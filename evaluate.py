"""
this code is an adaptation of the evaluation code of RGM:
Robust Point Cloud Registration Framework Based on Deep Graph Matching
https://github.com/fukexue/RGM/

"""

import torch
import time
from datetime import datetime
from pathlib import Path
import numpy as np

from rgm.data.data_loader import get_dataloader, get_datasets
from rgm.utils.config import cfg
from rgm.utils.hungarian import hungarian
from rgm.utils.loss_func import PermLoss
from rgm.utils.evaluation_metric import matching_accuracy, calcorrespondpc, square_distance, to_numpy
from rgm.models.correspondSlover import SVDslover, RANSACSVDslover
from collections import defaultdict
from rgm.utils import dcputil
from rgm.utils.se3 import transform

from gafar.model import GAFAR
from gafar.misc import scores_to_prediction, estimate_transform


def compute_metrics(s_perm_mat, P1_gt, P2_gt, R_gt, T_gt, R_pre, T_pre, viz=None, usepgm=True, userefine=False,
                    recall_threshold_rot: float = 1.0, recall_threshold_trans: float = 0.1):
    # compute r,t
    # R_pre, T_pre = compute_transform(s_perm_mat, P1_gt, P2_gt, R_gt, T_gt, viz=viz, usepgm=usepgm, userefine=userefine)

    r_pre_euler_deg = dcputil.npmat2euler(R_pre.detach().cpu().numpy(), seq='xyz')
    r_gt_euler_deg = dcputil.npmat2euler(R_gt.detach().cpu().numpy(), seq='xyz')
    r_mse = np.mean((r_gt_euler_deg - r_pre_euler_deg) ** 2, axis=1)
    r_mae = np.mean(np.abs(r_gt_euler_deg - r_pre_euler_deg), axis=1)
    t_mse = torch.mean((T_gt - T_pre) ** 2, dim=1)
    t_mae = torch.mean(torch.abs(T_gt - T_pre), dim=1)

    # Rotation, translation errors (isotropic, i.e. doesn't depend on error
    # direction, which is more representative of the actual error)
    concatenated = dcputil.concatenate(dcputil.inverse(R_gt.cpu().numpy(), T_gt.cpu().numpy()),
                                       np.concatenate([R_pre.cpu().numpy(), T_pre.unsqueeze(-1).cpu().numpy()],
                                                      axis=-1))
    rot_trace = concatenated[:, 0, 0] + concatenated[:, 1, 1] + concatenated[:, 2, 2]
    residual_rotdeg = torch.acos(torch.clamp(0.5 * (rot_trace - 1), min=-1.0, max=1.0)) * 180.0 / np.pi
    residual_transmag = concatenated[:, :, 3].norm(dim=-1)

    recall = torch.logical_and(residual_rotdeg < recall_threshold_rot, residual_transmag < recall_threshold_trans)

    # Chamfer distance
    # src_transformed = transform(pred_transforms, points_src)
    P1_transformed = torch.from_numpy(transform(torch.cat((R_pre, T_pre[:, :, None]), dim=2).detach().cpu().numpy(),
                                                P1_gt.detach().cpu().numpy())).to(P1_gt)
    dist_src = torch.min(square_distance(P1_transformed, P2_gt), dim=-1)[0]
    dist_ref = torch.min(square_distance(P2_gt, P1_transformed), dim=-1)[0]
    chamfer_dist = torch.mean(dist_src, dim=1) + torch.mean(dist_ref, dim=1)

    # Source distance
    P1_pre_trans = torch.from_numpy(transform(torch.cat((R_pre, T_pre[:, :, None]), dim=2).detach().cpu().numpy(),
                                              P1_gt.detach().cpu().numpy())).to(P1_gt)
    P1_gt_trans = torch.from_numpy(transform(torch.cat((R_gt, T_gt[:, :, None]), dim=2).detach().cpu().numpy(),
                                             P1_gt.detach().cpu().numpy())).to(P1_gt)
    dist_src = torch.min(square_distance(P1_pre_trans, P1_gt_trans), dim=-1)[0]
    presrc_dist = torch.mean(dist_src, dim=1)

    # Clip Chamfer distance
    clip_val = torch.Tensor([0.1]).cuda()
    P1_transformed = torch.from_numpy(transform(torch.cat((R_pre, T_pre[:, :, None]), dim=2).detach().cpu().numpy(),
                                                P1_gt.detach().cpu().numpy())).to(P1_gt)
    dist_src = torch.min(torch.min(torch.sqrt(square_distance(P1_transformed, P2_gt)), dim=-1)[0], clip_val)
    dist_ref = torch.min(torch.min(torch.sqrt(square_distance(P2_gt, P1_transformed)), dim=-1)[0], clip_val)
    clip_chamfer_dist = torch.mean(dist_src, dim=1) + torch.mean(dist_ref, dim=1)

    # correspondence distance
    P2_gt_copy, _ = calcorrespondpc(s_perm_mat, P2_gt.detach())
    inlier_src = torch.sum(s_perm_mat, axis=-1)[:, :, None]
    # inlier_ref = torch.sum(s_perm_mat, axis=-2)[:, :, None]
    P1_gt_trans_corr = P1_gt_trans.mul(inlier_src)
    P2_gt_copy_coor = P2_gt_copy.mul(inlier_src)
    correspond_dis=torch.sqrt(torch.sum((P1_gt_trans_corr-P2_gt_copy_coor)**2, dim=-1, keepdim=True))
    correspond_dis[inlier_src == 0] = np.nan

    metrics = {'r_mse': r_mse,
               'r_mae': r_mae,
               't_mse': to_numpy(t_mse),
               't_mae': to_numpy(t_mae),
               'err_r_deg': to_numpy(residual_rotdeg),
               'err_t': to_numpy(residual_transmag),
               'chamfer_dist': to_numpy(chamfer_dist),
               'pcab_dist': to_numpy(presrc_dist),
               'clip_chamfer_dist': to_numpy(clip_chamfer_dist),
               'pre_transform':np.concatenate((to_numpy(R_pre),to_numpy(T_pre)[:,:,None]),axis=2),
               'gt_transform':np.concatenate((to_numpy(R_gt),to_numpy(T_gt)[:,:,None]),axis=2),
               'cpd_dis_nomean':to_numpy(correspond_dis),
               'recall': to_numpy(recall)}

    return metrics


def summarize_metrics(metrics):
    """Summaries computed metrices by taking mean over all data instances"""
    summarized = {}
    for k in metrics:
        if k.endswith('mse'):
            summarized[k[:-3] + 'rmse'] = np.sqrt(np.mean(metrics[k]))
        elif k.startswith('err') or k.startswith('valid_err'):
            summarized[k + '_mean'] = np.mean(metrics[k])
            summarized[k + '_rmse'] = np.sqrt(np.mean(metrics[k]**2))
        elif k.endswith('nomean'):
            summarized[k] = metrics[k]
        else:
            summarized[k] = np.mean(metrics[k])

    return summarized


def print_metrics(summary_metrics: dict, title: str = 'Metrics', prepends: list[str] = ['']):
    """Prints out formated metrics to logger"""

    print('=' * (len(title) + 1))
    print(title + ':')

    for val in prepends:
        if val != '':
            print(val)
        print('DeepCP metrics:{:.4f}(rot-rmse) | {:.4f}(rot-mae) | {:.4g}(trans-rmse) | {:.4g}(trans-mae)'.
              format(summary_metrics[val + 'r_rmse'], summary_metrics[val + 'r_mae'],
                     summary_metrics[val + 't_rmse'], summary_metrics[val + 't_mae'],
                     ))
        print('Rotation error {:.4f}(deg, mean) | {:.4f}(deg, rmse)'.
              format(summary_metrics[val + 'err_r_deg_mean'],
                     summary_metrics[val + 'err_r_deg_rmse']))
        print('Translation error {:.4g}(mean) | {:.4g}(rmse)'.
              format(summary_metrics[val + 'err_t_mean'],
                     summary_metrics[val + 'err_t_rmse']))
        print('RPM Chamfer error: {:.7f}(mean-sq)'.
              format(summary_metrics[val + 'chamfer_dist']))
        print('Source error: {:.7f}(mean-sq)'.
              format(summary_metrics[val + 'pcab_dist']))
        print('Clip Chamfer error: {:.7f}(mean-sq)'.
              format(summary_metrics[val + 'clip_chamfer_dist']))


def predict_correspondences(
        source: torch.Tensor,
        reference: torch.Tensor,
        model: torch.nn.Module,
        threshold: float = 0.2,
) -> [dict, torch.Tensor, torch.Tensor]:
    rotation_nn, translation_nn = None, None
    estimate = model(source, reference)

    prediction = scores_to_prediction(estimate['scores'].detach(), threshold=threshold)

    if 'rotation_estimate' in estimate:
        rotation_nn, translation_nn = estimate['rotation_estimate'], estimate['translation_estimate']
    # else:
    #     rotation_nn, translation_nn = weighted_transform_from_score(
    #         source,
    #         reference,
    #         estimate['scores'].detach().double()
    #     )

    return prediction, rotation_nn, translation_nn


def eval_model(model, dataloader, eval_epoch=None, metric_is_save=False, estimate_iters=1,
               viz=None, usepgm=True, userefine=False, save_filetime='time', threshold: float = 0.5):
    print('-----------------Start evaluation-----------------')
    lap_solver = hungarian
    permevalLoss = PermLoss()
    since = time.time()
    prediction_time = 0.0
    all_val_metrics_np = [defaultdict(list) for _ in range(estimate_iters)]
    iter_num = 0

    dataset_size = len(dataloader.dataset)
    print('train datasize: {}'.format(dataset_size))
    device = next(model.parameters()).device
    print('model on device: {}'.format(device))
    print(f'threshold: {threshold}')

    was_training = model.training
    model.eval()
    running_since = time.time()

    print(f'number of registration iterations: {estimate_iters}')
    save_batch = False
    for inputs in dataloader:
        P1_gt, P2_gt = [_.cuda() for _ in inputs['Ps']]
        n1_gt, n2_gt = [_.cuda() for _ in inputs['ns']]
        perm_mat = inputs['gt_perm_mat'].cuda()
        T1_gt, T2_gt = [_.cuda() for _ in inputs['Ts']]
        Inlier_src_gt, Inlier_ref_gt = [_.cuda() for _ in inputs['Ins']]
        Label = torch.tensor([_ for _ in inputs['label']])

        batch_cur_size = perm_mat.size(0)
        iter_num = iter_num + 1
        infer_time = time.time()

        o3d = None
        if save_batch:
            import open3d as o3d
            for batch_idx in range(batch_cur_size):
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(P1_gt[batch_idx, :, :3].cpu().numpy())
                pcd.normals = o3d.utility.Vector3dVector(P1_gt[batch_idx, :, 3:6].cpu().numpy())
                name = f'cloud_{batch_idx}_source.ply'
                path = Path(cfg.OUTPUT_PATH) / name
                o3d.io.write_point_cloud(str(path), pcd)

                pcd.points = o3d.utility.Vector3dVector(P2_gt[batch_idx, :, :3].cpu().numpy())
                pcd.normals = o3d.utility.Vector3dVector(P2_gt[batch_idx, :, 3:6].cpu().numpy())
                name = f'cloud_{batch_idx}_reference.ply'
                path = Path(cfg.OUTPUT_PATH) / name
                o3d.io.write_point_cloud(str(path), pcd)

        source_aligned = P1_gt
        with torch.no_grad():
            for num in range(estimate_iters):
                prediction_time -= time.time()
                prediction, rotation_nn, translation_nn = predict_correspondences(
                    source_aligned,
                    P2_gt,
                    model,
                    threshold=threshold,
                )

                rotation, translation = estimate_transform(P1_gt[:, :, :3], P2_gt[:, :, :3], prediction)

                source_aligned = torch.cat(
                    (torch.matmul(P1_gt[..., :3], rotation.float().transpose(2, 1)) + translation.float().view(-1, 1, 3),
                     torch.matmul(P1_gt[..., 3:6], rotation.float().transpose(2, 1))), dim=2)

                prediction_time += time.time()

                permevalloss = torch.tensor([0])

                if save_batch:
                    for batch_idx in range(batch_cur_size):
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(source_aligned[batch_idx, :, :3].cpu().numpy())
                        pcd.normals = o3d.utility.Vector3dVector(source_aligned[batch_idx, :, 3:6].cpu().numpy())
                        name = f'cloud_{batch_idx}_source_aligned_{num}.ply'
                        path = Path(cfg.OUTPUT_PATH) / name
                        o3d.io.write_point_cloud(str(path), pcd)

                # create permutation matrix from PSR predictions
                s_perm_mat = torch.zeros_like(perm_mat)
                ref_range = torch.arange(n2_gt.max(), dtype=torch.long)
                valid = prediction['matches1'] >= 0
                for b in range(s_perm_mat.shape[0]):
                    s_perm_mat[b, prediction['matches1'][b, valid[b]], ref_range[valid[b]]] = 1

                infer_time = time.time() - infer_time
                match_metrics = matching_accuracy(s_perm_mat, perm_mat, n1_gt)
                perform_metrics = compute_metrics(s_perm_mat, P1_gt[:,:,:3], P2_gt[:,:,:3], T1_gt[:,:3,:3], T1_gt[:,:3,3],
                                                  rotation, translation.view(batch_cur_size, 3))

                valid = (valid.sum(1) > 2).cpu().numpy()
                for k in match_metrics:
                    all_val_metrics_np[num][k].append(match_metrics[k])
                    all_val_metrics_np[num]['valid_' + k].append(match_metrics[k][valid])
                for k in perform_metrics:
                    all_val_metrics_np[num][k].append(perform_metrics[k])
                    all_val_metrics_np[num]['valid_' + k].append(perform_metrics[k][valid])
                all_val_metrics_np[num]['valid_nomean'].append(valid)
                all_val_metrics_np[num]['label'].append(Label)
                all_val_metrics_np[num]['loss'].append(np.repeat(permevalloss.item(), valid.sum().item()))
                all_val_metrics_np[num]['infertime'].append(np.repeat(infer_time/batch_cur_size, batch_cur_size))

        save_batch = False

        if iter_num % cfg.STATISTIC_STEP == 0 and metric_is_save:
            running_speed = cfg.STATISTIC_STEP * batch_cur_size / (time.time() - running_since)
            print('Iteration {:<4} {:>4.2f}sample/s'.format(iter_num, running_speed))
            running_since = time.time()

    all_val_metrics_np = [{k: np.concatenate(val_metrics[k]) for k in val_metrics} for val_metrics in all_val_metrics_np]
    for num, val_metrics in enumerate(all_val_metrics_np):
        print(f'iteration {num} (valid: {val_metrics["valid_nomean"].sum()}[{val_metrics["valid_nomean"].mean():.2%}]):')
        summary_metrics = summarize_metrics(val_metrics)
        print('Mean-Loss: {:.4f} GT-Acc:{:.4f} Pred-Acc:{:.4f}'.format(summary_metrics['loss'], summary_metrics['acc_gt'], summary_metrics['acc_pred']))
        print(f'recall: {summary_metrics["recall"]:5.2%}')
        print(f'recall valid: {summary_metrics["valid_recall"]:5.2%}')
        print_metrics(summary_metrics, prepends=['', 'valid_'])
        if metric_is_save:
            np.save(str(Path(args.output) / ('eval_log_' + save_filetime + '_metric')),
                    all_val_metrics_np)

        print()

    time_elapsed = time.time() - since
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(f'Registrations/second: {dataset_size/time_elapsed}')
    print(f'prediction only Registrations/second {dataset_size/prediction_time}')
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable parameters: {pytorch_total_params}')

    model.train(mode=was_training)

    return summary_metrics


def caliters_perm(model, P1_gt_copy, P2_gt_copy, A1_gt, A2_gt, n1_gt, n2_gt, estimate_iters):
    lap_solver1 = hungarian
    s_perm_indexs = []
    for estimate_iter in range(estimate_iters):
        s_prem_i, Inlier_src_pre, Inlier_ref_pre = model(P1_gt_copy, P2_gt_copy,
                                                         A1_gt, A2_gt, n1_gt, n2_gt)
        if cfg.PGM.USEINLIERRATE:
            s_prem_i = Inlier_src_pre * s_prem_i * Inlier_ref_pre.transpose(2, 1).contiguous()
        s_perm_i_mat = lap_solver1(s_prem_i, n1_gt, n2_gt, Inlier_src_pre, Inlier_ref_pre)
        P2_gt_copy1, s_perm_i_mat_index = calcorrespondpc(s_perm_i_mat, P2_gt_copy)
        s_perm_indexs.append(s_perm_i_mat_index)
        if cfg.EXPERIMENT.USERANSAC:
            R_pre, T_pre, s_perm_i_mat = RANSACSVDslover(P1_gt_copy[:,:,:3], P2_gt_copy1[:,:,:3], s_perm_i_mat)
        else:
            R_pre, T_pre = SVDslover(P1_gt_copy[:,:,:3], P2_gt_copy1[:,:,:3], s_perm_i_mat)
        P1_gt_copy[:,:,:3] = torch.bmm(P1_gt_copy[:,:,:3], R_pre.transpose(2, 1).contiguous()) + T_pre[:, None, :]
        P1_gt_copy[:,:,3:6] = P1_gt_copy[:,:,3:6] @ R_pre.transpose(-1, -2)
    return s_perm_i_mat


if __name__ == '__main__':
    from rgm.utils.dup_stdout_manager import DupStdoutFileManager
    from rgm.utils.parse_argspc import parse_args
    from rgm.utils.print_easydict import print_easydict

    import scipy
    if np.__version__ == '1.19.2' and scipy.__version__ == '1.5.0':
        print('It is the same as the paper result')

    args = parse_args('Point could registration of graph matching evaluation code.')

    torch.manual_seed(cfg.RANDOM_SEED)

    pc_dataset = get_datasets(partition = 'test',
                              num_points = cfg.DATASET.POINT_NUM,
                              unseen = cfg.DATASET.UNSEEN,
                              noise_type = cfg.DATASET.NOISE_TYPE,
                              rot_mag = cfg.DATASET.ROT_MAG,
                              trans_mag = cfg.DATASET.TRANS_MAG,
                              partial_p_keep = cfg.DATASET.PARTIAL_P_KEEP,
                              non_deterministic_test=args.non_deterministic)

    dataloader = get_dataloader(pc_dataset)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.model)
    if hasattr(ckpt['model'], 'config'):
        model = GAFAR(ckpt['model'].config)
        model.load_state_dict(ckpt['model'].state_dict())
    else:
        model = GAFAR(ckpt['config'])
        model.load_state_dict(ckpt['model'])
    model = model.to(device)
    # model = DataParallel(model, device_ids=cfg.GPUS)

    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)
    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    with DupStdoutFileManager(str(Path(args.output) / ('eval_log_' + now_time + '.log'))) as _:
        print_easydict(cfg)
        print(f'testing non-deterministically: {"YES" if args.non_deterministic else "NO"}')
        metrics = eval_model(model, dataloader,
                             eval_epoch=cfg.EVAL.EPOCH if cfg.EVAL.EPOCH != 0 else None,
                             metric_is_save=True,
                             estimate_iters=cfg.EVAL.ITERATION_NUM,
                             usepgm=cfg.EXPERIMENT.USEPGM, userefine=cfg.EXPERIMENT.USEREFINE,
                             save_filetime=now_time,
                             threshold=args.threshold)
