"""
evaluate GAFARv2 on Kitti Odometry Benchmark, 3DMatch and 3DLoMatch dataset

this code uses copies and adaptions of testing/validation source code of GeoTransformer:
https://github.com/qinzheng93/GeoTransformer

"""
import torch
import numpy as np
import open3d as o3d

from pathlib import Path
from argparse import ArgumentParser, Namespace
from easydict import EasyDict as edict
from collections import defaultdict
from tqdm import tqdm

# put GeoTransformer into search path
from geot.utils.summary_board import SummaryBoard

# from GeoTransformer.geotransformer.utils.summary_board import SummaryBoard
from geot.datasets.registration.threedmatch import ThreeDMatchPairDataset
from geot.datasets.registration.kitti import OdometryKittiPairDataset
from geot.utils.common import get_log_string
from geot.utils.timer import TimerGAFAR as Timer
from geot.utils.torch import to_cuda, release_cuda
from geot.utils.loss import Evaluator
from geot.utils.datasets import train_valid_data_loader, test_data_loader

from gafar import GAFAR, GAFARv2
from gafar.misc import scores_to_prediction, estimate_transform
from utils.logs import get_logger
from utils.random_state import rng_save_restore
from utils.robust import irls_sigmoid_single


########################################################################################################################
# GeoTransformer default dataset configs - copy as standard dictionary
cfg = edict()

#
# 3DMatch/3DLoMatch
#
cfg['3DMatch'] = edict()
cfg['3DMatch'].data = edict()
cfg['3DMatch'].data.dataset = ThreeDMatchPairDataset
cfg['3DMatch'].data.dataset_root = 'set me via --dataset flag'
cfg['3DMatch'].data.split = {
    'test': '3DMatch', 'train': 'train', 'val': 'val', '3DMatch': '3DMatch', '3DLoMatch': '3DLoMatch'}
# train data
cfg['3DMatch'].train = edict()
cfg['3DMatch'].train.batch_size = 1
cfg['3DMatch'].train.num_workers = 8
cfg['3DMatch'].train.point_limit = 30000
cfg['3DMatch'].train.use_augmentation = True
cfg['3DMatch'].train.augmentation_noise = 0.005
cfg['3DMatch'].train.augmentation_rotation = 1.0
# test data
cfg['3DMatch'].test = edict()
cfg['3DMatch'].test.batch_size = 1
cfg['3DMatch'].test.num_workers = 8
cfg['3DMatch'].test.point_limit = None
# evaluation
cfg['3DMatch'].eval = edict()
cfg['3DMatch'].eval.acceptance_overlap = 0.0
cfg['3DMatch'].eval.acceptance_radius = 0.1
cfg['3DMatch'].eval.inlier_ratio_threshold = 0.05
cfg['3DMatch'].eval.rmse_threshold = 0.2
cfg['3DMatch'].eval.rre_threshold = 15.0
cfg['3DMatch'].eval.rte_threshold = 0.3
# normals
cfg['3DMatch'].normals = edict()
cfg['3DMatch'].normals.nn = 30
cfg['3DMatch'].normals.distance = 0.5
# icp
cfg['3DMatch'].icp = edict()
cfg['3DMatch'].icp.inlier_distance = 0.1
# irls
cfg['3DMatch'].irls = edict()
cfg['3DMatch'].irls.inlier_distance = 0.1

#
# Kitti
#
cfg['Kitti'] = edict()
cfg['Kitti'].data = edict()
cfg['Kitti'].data.dataset = OdometryKittiPairDataset
cfg['Kitti'].data.dataset_root = 'set me via --dataset flag'
cfg['Kitti'].data.split = {'test': 'test', 'train': 'train', 'val': 'val'}
# train data
cfg['Kitti'].train = edict()
cfg['Kitti'].train.batch_size = 1
cfg['Kitti'].train.num_workers = 8
cfg['Kitti'].train.point_limit = 30000
cfg['Kitti'].train.use_augmentation = False
cfg['Kitti'].train.augmentation_noise = 0.01
cfg['Kitti'].train.augmentation_min_scale = 0.8
cfg['Kitti'].train.augmentation_max_scale = 1.2
cfg['Kitti'].train.augmentation_shift = 2.0
cfg['Kitti'].train.augmentation_rotation = 1.0
# test config
cfg['Kitti'].test = edict()
cfg['Kitti'].test.batch_size = 1
cfg['Kitti'].test.num_workers = 8
cfg['Kitti'].test.point_limit = None
# eval config
cfg['Kitti'].eval = edict()
cfg['Kitti'].eval.acceptance_overlap = 0.0
cfg['Kitti'].eval.acceptance_radius = 1.0
cfg['Kitti'].eval.inlier_ratio_threshold = 0.05
cfg['Kitti'].eval.rmse_threshold = 1.0
cfg['Kitti'].eval.rre_threshold = 5.0
cfg['Kitti'].eval.rte_threshold = 2.0
# normals
cfg['Kitti'].normals = edict()
cfg['Kitti'].normals.nn = 30
cfg['Kitti'].normals.distance = 4.75
# icp
cfg['Kitti'].icp = edict()
cfg['Kitti'].icp.inlier_distance = 0.3
# irls
cfg['Kitti'].irls = edict()
cfg['Kitti'].irls.inlier_distance = 0.75


########################################################################################################################
@torch.no_grad()
def main(opts: Namespace):
    logger = get_logger((opts.output / f'{opts.model.stem}_{opts.benchmark}_{opts.split}').with_suffix('.log'))
    logger.info(f'evaluating model {opts.model} on benchmark <{opts.benchmark}>')

    assert torch.cuda.is_available(), 'no cuda'
    device = torch.device('cuda')

    # load model
    ckpt = torch.load(opts.model)

    if hasattr(ckpt['model'], 'config'):
        model_config = ckpt['model'].config
    elif 'config' in ckpt:
        model_config = ckpt['config']
    else:
        raise RuntimeError('failed to load model')
    model = GAFARv2 if 'type' in model_config and model_config['type'].lower() == 'gafarv2' else GAFAR
    model = model(model_config)
    if hasattr(ckpt['model'], 'state_dict'):
        model.load_state_dict(ckpt['model'].state_dict())
    else:
        model.load_state_dict(ckpt['model'])

    model.to(device)
    model.eval()
    if opts.radius is not None:
        logger.info(f'updating neighbourhood search radius to {opts.radius:.3f}')
        model.update_radius(opts.radius)
    model_config = model.config

    logger.info(f'testing model of type \"{model_config["type"]}\"')
    logger.info(f'feature dimension <{model_config["feature_dimension"]}>, '
                f'matching attention <{model_config["matcher"]["attention"]}>'
                )

    data_config = cfg[opts.benchmark]
    data_config.data.dataset_root = opts.dataset
    summary_board = SummaryBoard(adaptive=True)
    summary_board_valid = SummaryBoard(adaptive=True)
    summary_board_icp = SummaryBoard(adaptive=True)
    summary_board_icp_valid = SummaryBoard(adaptive=True)
    summary_board_irls = SummaryBoard(adaptive=True)
    summary_board_irls_valid = SummaryBoard(adaptive=True)
    gt_evaluator = Evaluator(data_config)
    timer = Timer()

    # get dataset
    if opts.split not in ['train', 'val']:
        dataset = test_data_loader(data_config, data_config.data.split[opts.split])
    else:
        train_dataset, val_dataset = train_valid_data_loader(data_config)
        dataset = train_dataset if opts.split == 'train' else val_dataset

    logger.info(f'testing benchmark {opts.benchmark}, split {opts.split}')
    logger.info(f'threshold {opts.threshold}, minimum matches: {opts.min_matches}')
    logger.info(f'batch size {opts.batch_size}')

    logger.info(f'number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    logger.info(f'setting random seed: {opts.seed}')
    rng_save_restore(opts.seed)

    if opts.batch_size > 1:
        logger.warning('for proper timing set batch-size to 1!')

    icp_inlier_distance = None
    if not opts.no_icp:
        if opts.icp_distance is not None:
            icp_inlier_distance = opts.icp_distance
        else:
            icp_inlier_distance = data_config.icp.inlier_distance
        logger.info(f'running point-to-{opts.icp_method} ICP with inlier distance {icp_inlier_distance:.3f} and '
                    f'max iteration: {opts.icp_max_iteration}')

    irls_inlier_distance = None
    if not opts.no_irls:
        if opts.irls_distance is not None:
            irls_inlier_distance = opts.irls_distance
        else:
            irls_inlier_distance = data_config.irls.inlier_distance
        logger.info(f'running IRLS with inlier distance {irls_inlier_distance:.3f} '
                    f'for {opts.irls_max_iteration} iterations.')

    # main evaluation loop over data
    data = []
    full_data = defaultdict(list)
    num_valid = 0
    num_valid_irls = 0
    weight_stats = defaultdict(list)
    for batch_idx, item in tqdm(enumerate(dataset)):
        # aggregate one GAFAR batch

        # undo collation of GeoTransformer
        # points = item['points'][:, :3]
        # ref_length = item['lengths'][0].item()
        # ref_points = points[:ref_length]
        # src_points = points[ref_length:]
        ref_points = item['ref_points'].squeeze()
        src_points = item['src_points'].squeeze()

        # data subsampling and pre-processing
        if ref_points.shape[0] > opts.points:
            ref_indices = np.random.permutation(ref_points.shape[0])[:opts.points]
        else:
            ref_indices = np.arange(ref_points.shape[0])
        if src_points.shape[0] > opts.points:
            src_indices = np.random.permutation(src_points.shape[0])[:opts.points]
        else:
            src_indices = np.arange(src_points.shape[0])
        # sub sample/normals
        timer.tic('prepare')
        pcd_ref = o3d.t.geometry.PointCloud(
            o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(ref_points[:, :3].to(device))))
        pcd_ref.estimate_normals(radius=data_config.normals.distance, max_nn=data_config.normals.nn)
        full_data['ref_points_c'].append(torch.cat(
            (ref_points[ref_indices, :3].to(device),
             torch.from_dlpack(pcd_ref.point.normals.to_dlpack())[ref_indices].to(device) *
             torch.sign(torch.rand((ref_indices.shape[0], 1)) - 0.5).to(device)), 1))

        pcd_src = o3d.t.geometry.PointCloud(
            o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(src_points[:, :3].to(device))))
        pcd_src.estimate_normals(radius=data_config.normals.distance, max_nn=data_config.normals.nn)
        full_data['src_points_c'].append(torch.cat(
            (src_points[src_indices, :3].to(device),
             torch.from_dlpack(pcd_src.point.normals.to_dlpack())[src_indices].to(device) *
             torch.sign(torch.rand((ref_indices.shape[0], 1)) - 0.5).to(device)), 1))
        timer.toc('prepare')

        data.append({'transform': to_cuda(item['transform'])})
        full_data['ref_points'].append(ref_points.to(device))
        full_data['src_points'].append(src_points.to(device))

        # if batch not full and not end of dataset, continue
        if len(data) < opts.batch_size and batch_idx < (len(dataset) - 1):
            continue

        # run model
        source_points = torch.cat([x.unsqueeze(0) for x in full_data['src_points_c']], dim=0)
        reference_points = torch.cat([x.unsqueeze(0) for x in full_data['ref_points_c']], dim=0)

        timer.tic('model')

        scores = model(source_points, reference_points)
        prediction = scores_to_prediction(scores['score'].detach(),
                                          opts.threshold,
                                          discard_bin=opts.with_bin)

        rotation_estimate, translation_estimate, source_matching_reference = estimate_transform(
            source_points[:, :, :3].double(),
            reference_points[:, :, :3].double(),
            # predictions,
            {
                'matches1': prediction['matches1'].to(device),
                'matching_scores1': prediction['matching_scores1'].double().to(device),
            },
        )

        timer.toc('model')

        # evaluate batch
        batch_size = rotation_estimate.shape[0]
        valid = prediction['matches1'] >= 0
        transform = torch.zeros((batch_size, 4, 4), dtype=rotation_estimate.dtype, device=device)
        transform[:, :3, :3] = rotation_estimate
        transform[:, :3, 3] = translation_estimate
        transform[:, 3, 3] = 1.0
        for idx in range(rotation_estimate.shape[0]):

            output_dict = {
                'ref_corr_points': reference_points[idx][valid[idx], :3],
                'src_corr_points': source_matching_reference[idx][valid[idx], :3],
                'ref_points_c': full_data['ref_points_c'][idx],
                'src_points_c': full_data['src_points_c'][idx],
                'ref_points': full_data['ref_points'][idx],
                'src_points': full_data['src_points'][idx],
                'corr_scores': prediction['matching_scores1'][idx, valid[idx]],
                'estimated_transform': transform[idx],
            }

            result = release_cuda(gt_evaluator(output_dict, data[idx]))
            summary_board.update_from_result_dict(result)
            if valid[idx].sum() > opts.min_matches:
                num_valid += 1
                summary_board_valid.update_from_result_dict(result)

            if output_dict['corr_scores'].shape[0] > 0:
                weight_stats['min'].append(output_dict['corr_scores'].min().item())
                weight_stats['max'].append(output_dict['corr_scores'].max().item())
                weight_stats['mean'].append(output_dict['corr_scores'].mean().item())
            weight_stats['points'].append(output_dict['ref_corr_points'].shape[0])

            # perform iterative re-weighted least squares alignment on points
            if not opts.no_irls:
                timer.tic('irls')

                src_points = torch.gather(
                    output_dict['src_points_c'][:, :3], 0,
                    prediction['matches1'][idx, valid[idx]].unsqueeze(1).expand(-1, 3)).float().to(device)
                rotation, translation, weights = irls_sigmoid_single(
                    src_points, output_dict['ref_corr_points'].float().to(device),
                    rotation_estimate[idx].float().to(device), translation_estimate[idx].float().to(device),
                    irls_inlier_distance, iteration=opts.irls_max_iteration)

                timer.toc('irls')

                v_indces = torch.clone(valid[idx])
                valid[idx, v_indces] = weights > 0.01

                transform[idx, :3, :3] = rotation.to(device)
                transform[idx, :3, 3] = translation.to(device)

                output_dict['estimated_transform'] = transform[idx]
                result = release_cuda(gt_evaluator(output_dict, data[idx]))
                summary_board_irls.update_from_result_dict(result)
                if valid[idx].sum() > opts.min_matches:
                    num_valid_irls += 1
                    summary_board_irls_valid.update_from_result_dict(result)

            # perform ICP registration on top of network
            if not opts.no_icp:
                timer.tic('icp')
                if opts.batch_size > 1:
                    pcd_ref = o3d.t.geometry.PointCloud(o3d.core.Tensor.from_dlpack(
                        torch.utils.dlpack.to_dlpack(full_data['ref_points'][idx].to(device))))
                    if opts.icp_method == 'plane':
                        pcd_ref.estimate_normals(data_config.normals.nn, data_config.normals.distance)
                    pcd_src = o3d.t.geometry.PointCloud(o3d.core.Tensor.from_dlpack(
                        torch.utils.dlpack.to_dlpack(full_data['src_points'][idx].to(device))))

                if opts.icp_method == 'point':
                    method = o3d.t.pipelines.registration.TransformationEstimationPointToPoint()
                else:
                    method = o3d.t.pipelines.registration.TransformationEstimationPointToPlane()

                result_icp = o3d.t.pipelines.registration.icp(
                    pcd_src, pcd_ref, icp_inlier_distance,
                    o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(transform[idx].to(device))), method,
                    o3d.t.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=opts.icp_relative,
                        relative_rmse=opts.icp_relative,
                        max_iteration=opts.icp_max_iteration)
                )

                timer.toc('icp')

                output_dict['estimated_transform'] = torch.from_dlpack(result_icp.transformation.to_dlpack()).to(device)
                result = release_cuda(gt_evaluator(output_dict, data[idx]))
                summary_board_icp.update_from_result_dict(result)
                summary_board_icp.update('icp_fit', result_icp.fitness)
                summary_board_icp.update('icp_rmse', result_icp.inlier_rmse)
                summary_board_icp.update('icp_iter', result_icp.num_iterations)
                if valid[idx].sum() > opts.min_matches:
                    summary_board_icp_valid.update_from_result_dict(result)
                    summary_board_icp_valid.update('icp_fit', result_icp.fitness)
                    summary_board_icp_valid.update('icp_rmse', result_icp.inlier_rmse)
                    summary_board_icp_valid.update('icp_iter', result_icp.num_iterations)

        # reset batch
        data = []
        full_data.clear()

        mem_info = torch.cuda.mem_get_info()
        summary_board.update('mem', (mem_info[1] - mem_info[0]) / 1024**3)
        torch.cuda.empty_cache()

    # print results
    logger.info(f'threshold {opts.threshold}, minimum matches: {opts.min_matches}')
    logger.info(f'weights min: {np.mean(weight_stats["min"]):.3f}/{min(weight_stats["min"]):.3f}  '
                f'mean: {np.mean(weight_stats["mean"]):.3f}  '
                f'max: {np.mean(weight_stats["max"]):.3f}/{max(weight_stats["max"]):.3f}')
    logger.info(f'correspondences min: {min(weight_stats["points"])}  mean: {np.mean(weight_stats["points"]):.3f}  '
                f'max: {max(weight_stats["points"])}')

    summary_dict = summary_board.summary()
    message = get_log_string(result_dict=summary_dict, timer=timer)
    logger.info('full results:')
    logger.info(message)

    timer.count_time['model'] = timer.count_time['prepare']
    summary_board_valid.meter_dict['RR_T'].set_length(summary_board.meter_dict['RR_T'].get_length())
    summary_board_valid.meter_dict['RR_RMSE'].set_length(summary_board.meter_dict['RR_RMSE'].get_length())
    message = get_log_string(result_dict=summary_board_valid.summary(), timer=timer)
    logger.info(f'valid results: {num_valid/len(dataset):.2%}({num_valid}/{len(dataset)})')
    logger.info(message)

    if not opts.no_irls:
        summary_dict = summary_board_irls.summary()
        message = get_log_string(result_dict=summary_dict)
        logger.info('IRLS full results:')
        logger.info(message)

        summary_board_irls_valid.meter_dict['RR_T'].set_length(summary_board_irls.meter_dict['RR_T'].get_length())
        summary_board_irls_valid.meter_dict['RR_RMSE'].set_length(summary_board_irls.meter_dict['RR_RMSE'].get_length())
        message = get_log_string(result_dict=summary_board_irls_valid.summary())
        logger.info(f'IRLS valid results: {num_valid_irls/len(dataset):.2%}({num_valid_irls}/{len(dataset)})')
        logger.info(message)

        num_valid = num_valid_irls

    if not opts.no_icp:
        summary_dict = summary_board_icp.summary()
        message = get_log_string(result_dict=summary_dict)
        logger.info('ICP full results:')
        logger.info(message)

        summary_board_icp_valid.meter_dict['RR_T'].set_length(summary_board_icp.meter_dict['RR_T'].get_length())
        summary_board_icp_valid.meter_dict['RR_RMSE'].set_length(summary_board_icp.meter_dict['RR_RMSE'].get_length())
        message = get_log_string(result_dict=summary_board_icp_valid.summary())
        logger.info(f'ICP valid results: {num_valid/len(dataset):.2%}({num_valid}/{len(dataset)})')
        logger.info(message)


########################################################################################################################
def parse_args() -> Namespace:
    parser = ArgumentParser(description='Evaluate GAFARv2 point cloud matching modules using GeoTransformer Dataset and'
                                        'Metric implementations.')
    parser.add_argument('model', metavar='gafar_model.pt', type=Path, help='Trained GAFARv2 model checkpoint.')
    parser.add_argument('output', metavar='path/to/output/', type=Path, help='Where to put log files etc.')
    parser.add_argument('--benchmark', type=str, choices=['3DMatch', '3DLoMatch', 'Kitti'],
                        help='Dataset/mode to test on.')
    parser.add_argument('-d', '--dataset', type=Path, default=None, help='Root of chosen GeoTransformer type dataset.')

    parser.add_argument('--batch-size', type=int, default=18, help='Batch size in testing.')
    parser.add_argument('--points', type=int, default=1024, help='Number of points for sub-sampling of point cloud.')
    parser.add_argument('--radius', type=float, default=None, help='Override neighbourhood radius.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test', '3DMatch', '3DLoMatch'],
                        help='Test on a different split than testing.')
    parser.add_argument('--threshold', type=float, default=0.1, help='Matching score threshold.')
    parser.add_argument('--with-bin', action='store_false', help='Do not discard dust bin row/col.')
    parser.add_argument('--min-matches', type=int, default=30,
                        help='Minimum amount of matches to consider matching attempt valid.')

    # ICP
    parser.add_argument('--no-icp', action='store_true', help='Do not perform ICP registration after network.')
    parser.add_argument('--icp-method', type=str, default='plane', choices=['point', 'plane'],
                        help='Weather to perform point-to-point or point-to-plane ICP.')
    parser.add_argument('--icp-distance', type=float, default=None,
                        help='Override correspondence distance in ICP from command line.')
    parser.add_argument('--icp-max-iteration', type=int, default=20, help='Maximum number of ICP iterations.')
    parser.add_argument('--icp-relative', type=float, default=1e-6, help='ICP convergence relative fitness.')

    # IRLS
    parser.add_argument('--no-irls', action='store_true', help='Do not perform IRLS based fine-alignment.')
    parser.add_argument('--irls-distance', type=float, default=None,
                        help='Override correspondence distance for IRLS from command line.')
    parser.add_argument('--irls-max-iteration', type=int, default=1, help='Maximum number of IRLS iterations.')

    return parser.parse_args()


########################################################################################################################
if __name__ == '__main__':
    args = parse_args()
    main(args)
