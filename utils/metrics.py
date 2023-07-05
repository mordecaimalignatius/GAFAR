"""
more involved metrics for evaluating network performance
numpy or numpy/torch implementations

Ludwig Mohr
ludwig.mohr@icg.tugraz.at

"""

import torch
from logging import getLogger
try:
    from torch_cluster import knn
except ImportError:
    logger = getLogger(__name__)
    logger.debug('torch-cluster not available, falling back to \"direct\" method of nearest neighbour calculation')
    knn = None
from typing import Optional


################################################################################
def transformation_error(
        rotation_prediction: torch.Tensor,
        translation_prediction: torch.Tensor,
        rotation_true: torch.Tensor,
        translation_true: torch.Tensor,
        eps: torch.float64 = torch.finfo(torch.float64).eps,
) -> (torch.Tensor, torch.Tensor):
    """
    calculate angular error and translational error between estimated rigid transform and ground truth

    rotation error: angle of residual rotation inv(r_true) * r_pred, arccos((trace(r_res) -1) / 2)
    translation error: ||t_true - t_pred||_2

    :param rotation_prediction: predicted rotation matrices, Bx3x3
    :param translation_prediction: predicted translation vectors, Bx3
    :param rotation_true: true rotation matrices, Bx3x3
    :param translation_true: true translation vectors, Bx3
    :param eps: float precision to use for arccos clamping
    :returns: rotational error, Bx, translational error, Bx
    """
    f_max = 1. - eps

    matmul_state = torch.backends.cuda.matmul.allow_tf32
    if rotation_true.device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False

    r_res = torch.matmul(rotation_true.cpu().double().transpose(1, 2), rotation_prediction.cpu().double())
    e_rot = torch.arccos(torch.clamp((torch.einsum('jii->j', r_res) - 1.) / 2., min=-f_max, max=f_max))

    e_trans = torch.linalg.norm(translation_true.cpu().double() - translation_prediction.cpu().double(), dim=1)

    if rotation_true.device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = matmul_state

    return e_rot, e_trans


def evaluate_performance(
        prediction: dict,
        ground_truth: torch.Tensor,
        # device: torch.device = torch.device('cpu'),
) -> (torch.Tensor, torch.Tensor, dict):
    """
    calculate precision/recall metrics from SG match result

    :param prediction: output of superglue scores_to_prediction() (dict)
    :param ground_truth: ground truth matches source->target (BxN torch.Tensor)
    # :param device: torch device to use for calculations
    :returns:
    """
    device = ground_truth.device
    batch_size = ground_truth.shape[0]
    # get number of valid correspondences: correctly identified correspondences plus points in both, source and target,
    # correctly identified as not having a correspondence
    predictions_tgt = prediction['matches1'].detach().to(device)
    predictions_src = prediction['matches0'].detach().to(device)
    scores_tgt = prediction['matching_scores1'].detach().to(device)

    predictions_true = predictions_tgt == ground_truth
    predictions_true_sum = predictions_true.sum(1)

    positives_matched = ground_truth >= 0
    predictions_matched = predictions_tgt >= 0
    true_positives_matched = torch.logical_and(predictions_true, predictions_matched).sum(1)

    predictions_matched_sum = predictions_matched.sum(1)
    positives_matched_sum = positives_matched.sum(1)

    precision_matched = true_positives_matched / predictions_matched_sum
    recall_matched = true_positives_matched / positives_matched_sum

    overall_predictions = predictions_tgt.shape[1] + predictions_src.shape[1] - predictions_matched_sum
    overall_positives = predictions_tgt.shape[1] + predictions_src.shape[1] - positives_matched_sum

    wrong_score_max = 0.
    wrong_score_min = 1.
    wrong_score_avg = 0.

    # get number of predictions. to be fair we count: the number of points that were matched plus the number of points
    # in both source and target predictions that do not have matches and were correctly matched as unmatched
    unmatched_src = predictions_src == -1
    unmatched_src_truth = predictions_src.new_full(predictions_src.shape, 1, dtype=torch.bool)
    predictions_false = torch.logical_and(torch.logical_not(predictions_true), predictions_matched)
    predictions_false_sum = predictions_false.sum(1)
    for idx in range(batch_size):
        if predictions_false_sum[idx] > 0:
            wrong_score = scores_tgt[idx, predictions_false[idx]]
            wrong_score_max = max(wrong_score.max(), wrong_score_max)
            wrong_score_min = min(wrong_score.min(), wrong_score_min)
            wrong_score_avg += wrong_score.sum() / wrong_score.shape[0]

        # additional 'true' (marked as unmatched) matches of source points
        matched_in_source = ground_truth[idx, positives_matched[idx]].long()
        unmatched_src_truth[idx, matched_in_source] = torch.tensor(0, dtype=torch.bool)

    num_wrong_example = (predictions_false_sum > 0).sum().item()

    predictions_unmatched_source_true = torch.logical_and(unmatched_src_truth, unmatched_src).sum(1)

    overall_predictions_true = predictions_unmatched_source_true + predictions_true_sum

    precision_overall = overall_predictions_true / overall_predictions
    recall_overall = overall_predictions_true / overall_positives

    # get number of valid correspondences: those with matches plus unmatched ones in source and target
    precision_matched = torch.where(torch.isnan(precision_matched), precision_matched.new_tensor(0.), precision_matched)
    precision_overall = torch.where(torch.isnan(precision_overall), precision_overall.new_tensor(0.), precision_overall)

    return (
        {'precision': precision_overall, 'recall': recall_overall},
        {'precision': precision_matched, 'recall': recall_matched},
        {'max': wrong_score_max, 'min': wrong_score_min, 'avg': wrong_score_avg, 'num_wrong': num_wrong_example}
    )


################################################################################
def chamfer_distance(
        source: torch.Tensor,
        target: torch.Tensor,
        num_workers: int = 1,
        method: Optional[str] = None,
):
    """
    calculate the chamfer distance loss as evaluation metric according to
        RPM-Net: Robust Point Matching using Learned Features (Yew, Lee, CVPR 2020)

    :param source: transformed source points batch (BxNxF) or (NxF)
    :param target: target points batch (BxMxF) or (MxF)
    :param num_workers: Number of workers to use for computation. Has no
            effect in case :obj:`batch_x` or :obj:`batch_y` is not
            :obj:`None`, or the input lies on the GPU. (default: :obj:`1`)
    :param method: 'knn' (if torch_cluster.knn is available), 'direct' for linear algebra based greedy implementation.
            'direct' on gpu is fastest, 'knn' on cpu. for a large point sets (~10000) knn on cpu is faster than gpu
    :return: chamfer distance (Bx)
    """
    feature_dim = source.shape[-1]
    if source.shape[-1] != target.shape[-1]:
        raise RuntimeError('feature dimension of source and target point sets must be identical')

    if method is None:
        if source.device.type == 'cuda':
            method = 'direct'
        elif knn is not None:
            method = 'knn'
        else:
            method = 'direct'

    elif method == 'knn' and knn is None:
        method = 'direct'

    if method == 'knn':
        if source.ndim == 2:
            batch_x = None
            batch_y = None

            num_source = source.shape[0]
            num_target = target.shape[0]

            source_view = source
            target_view = target

            batch_size = 0

        else:
            if source.shape[0] != target.shape[0]:
                raise RuntimeError('batch sizes of source and target point sets must be identical')

            device = source.device
            batch_size = source.shape[0]
            num_source = source.shape[1]
            num_target = target.shape[1]
            batch_x = torch.arange(
                batch_size, dtype=torch.long, device=device).unsqueeze(1).expand(-1, num_source).view(-1)
            batch_y = torch.arange(
                batch_size, dtype=torch.long, device=device).unsqueeze(1).expand(-1, num_target).view(-1)

            source_view = source.view(-1, source.shape[2])
            target_view = target.view(-1, target.shape[2])

        index = knn(source_view, target_view, k=1, batch_x=batch_x, batch_y=batch_y, num_workers=num_workers)
        l2_sq = (target_view - source_view[index[1, :]]).view(-1, num_target, feature_dim)
        l2_sq = torch.einsum('bnk,bnk->bn', l2_sq, l2_sq)
        distance = torch.mean(l2_sq, dim=1)

        index = knn(target_view, source_view, k=1, batch_x=batch_y, batch_y=batch_x, num_workers=num_workers)
        l2_sq = (source_view - target_view[index[1, :]]).view(-1, num_source, feature_dim)
        l2_sq = torch.einsum('bnk,bnk->bn', l2_sq, l2_sq)
        distance += torch.mean(l2_sq, dim=1)

        if batch_size == 0:
            distance = distance.view(-1)

        return distance

    else:
        # linear algebra greedy nearest neighbours
        if source.ndim == 2:
            source = source.unsqueeze(0)
            target = target.unsqueeze(0)

        inner = 2 * torch.matmul(source, target.transpose(1, 2))
        source_2 = torch.einsum('bnd,bnd->bn', source, source).unsqueeze(2)
        target_2 = torch.einsum('bnd,bnd->bn', target, target).unsqueeze(1)

        # -(a-b)**2 = - a**2 + 2ab - b**2
        distance = inner - source_2 - target_2

        cd = -torch.mean(distance.topk(k=1, dim=1)[0].squeeze(1), dim=1) \
             - torch.mean(distance.topk(k=1, dim=2)[0].squeeze(2), dim=1)

        return cd


def chamfer_distance_modified(
        source: torch.Tensor,
        target: torch.Tensor,
        source_clean: torch.Tensor,
        target_clean: torch.Tensor,
        num_workers: int = 1,
        method: Optional[str] = None,
):
    """
    calculate the modified chamfer distance loss as evaluation metric according to
        RPM-Net: Robust Point Matching using Learned Features (Yew, Lee, CVPR 2020)

    compares the average closest distance to the *clean and full* point set

    :param source: transformed source points batch (BxNxF)
    :param target: target points batch (BxMxF)
    :param source_clean: transformed full and clean source points batch (BxNfxF)
    :param target_clean: full and clean target points batch (BxMfxF)
    :param num_workers: Number of workers to use for computation. Has no
            effect in case :obj:`batch_x` or :obj:`batch_y` is not
            :obj:`None`, or the input lies on the GPU. (default: :obj:`1`)
    :param method: 'knn' (if torch_cluster.knn is available), 'direct' for linear algebra based greedy implementation.
            'direct' on gpu is fastest, 'knn' on cpu. for a large point sets (~10000) knn on cpu is faster than gpu
    :return: modified chamfer distance (Bx)
    """
    feature_dim = source.shape[-1]
    if target.shape[-1] != feature_dim or source_clean.shape[-1] != feature_dim or \
            target_clean.shape[-1] != feature_dim:
        raise RuntimeError('feature dimension of source and target point sets must be identical')

    if method is None:
        if source.device.type == 'cuda':
            method = 'direct'
        elif knn is not None:
            method = 'knn'
        else:
            method = 'direct'

    elif method == 'knn' and knn is None:
        method = 'direct'

    if method == 'knn':
        if source.ndim == 2:
            batch_x = None
            batch_y = None
            batch_clean_x = None
            batch_clean_y = None

            num_source = source.shape[0]
            num_target = target.shape[0]

            source_view = source
            target_view = target
            source_clean_view = source_clean
            target_clean_view = target_clean

            batch_size = 0

        else:
            batch_size = source.shape[0]
            if target.shape[0] != batch_size or source_clean.shape[0] != batch_size or target_clean.shape[0] != batch_size:
                raise RuntimeError('batch sizes of source and target point sets must be identical')

            device = source.device
            batch_x = torch.arange(
                batch_size, dtype=torch.long, device=device).unsqueeze(1).expand(-1, source.shape[1]).flatten()
            batch_y = torch.arange(
                batch_size, dtype=torch.long, device=device).unsqueeze(1).expand(-1, target.shape[1]).flatten()
            batch_clean_x = torch.arange(
                batch_size, dtype=torch.long, device=device).unsqueeze(1).expand(-1, source_clean.shape[1]).flatten()
            batch_clean_y = torch.arange(
                batch_size, dtype=torch.long, device=device).unsqueeze(1).expand(-1, target_clean.shape[1]).flatten()

            num_source = source.shape[1]
            num_target = target.shape[1]

            source_view = source.view(-1, source.shape[2])
            target_view = target.view(-1, target.shape[2])
            source_clean_view = source_clean.view(-1, source_clean.shape[2])
            target_clean_view = target_clean.view(-1, target_clean.shape[2])

        index = knn(
            target_clean_view, source_view, batch_x=batch_clean_y, batch_y=batch_x, k=1, num_workers=num_workers)
        l2_sq = (target_clean_view[index[1, :]] - source_view).view(-1, num_source, feature_dim)
        l2_sq = torch.einsum('bki,bki->bk', l2_sq, l2_sq)
        distance = torch.mean(l2_sq, dim=1)

        index = knn(
            source_clean_view, target_view, batch_x=batch_clean_x, batch_y=batch_y, k=1, num_workers=num_workers)
        l2_sq = (source_clean_view[index[1, :]] - target_view).view(-1, num_target, feature_dim)
        l2_sq = torch.einsum('bki,bki->bk', l2_sq, l2_sq)
        distance += torch.mean(l2_sq, dim=1)

        if batch_size == 0:
            distance = distance.view(-1)

        return distance

    else:
        # linear algebra greedy nearest neighbours
        if source.ndim == 2:
            source = source.unsqueeze(0)
            target = target.unsqueeze(0)
            source_clean = source_clean.unsqueeze(0)
            target_clean = target_clean.unsqueeze(0)

        inner = 2 * torch.matmul(target, source_clean.transpose(1, 2))  # BxMxD * BxDxN -> BxMxN
        source_2 = torch.einsum('bnd,bnd->bn', source_clean, source_clean).unsqueeze(1)     # Bx1xN
        target_2 = torch.einsum('bnd,bnd->bn', target, target).unsqueeze(2)     # BxMx1
        distance = inner - source_2 - target_2
        cd = -torch.mean(distance.topk(k=1, dim=2)[0].squeeze(2), dim=1)

        inner = 2 * torch.matmul(source, target_clean.transpose(1, 2))  # BxNxD * BxMxD -> BxNxM
        source_2 = torch.einsum('bnd,bnd->bn', source, source).unsqueeze(2)     # BxNx1
        target_2 = torch.einsum('bnd,bnd->bn', target_clean, target_clean).unsqueeze(1)     # Bx1xM
        distance = inner - source_2 - target_2
        cd -= torch.mean(distance.topk(k=1, dim=2)[0].squeeze(2), dim=1)

        return cd
