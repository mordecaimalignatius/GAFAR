"""
miscellaneous methods and classes for point set registration networks

Ludwig Mohr
ludwig.mohr@icg.tugraz.at

"""
import torch
from math import log, exp
from typing import Union


################################################################################
def normalize_3d_points_sphere(points_0, points_1):
    """ normalize 3D point clouds jointly to sit centered within the unit-sphere """
    points_0_mean = torch.mean(points_0, dim=1, keepdim=True)
    points_1_mean = torch.mean(points_1, dim=1, keepdim=True)
    points_0_centered = points_0 - points_0_mean
    points_1_centered = points_1 - points_1_mean
    points_length = 1. / torch.norm(torch.cat((points_0_centered, points_1_centered), dim=1), dim=2).max(1).values

    points_0_centered *= points_length.view(-1, 1, 1)
    points_1_centered *= points_length.view(-1, 1, 1)

    return points_0_centered, points_1_centered, points_length


################################################################################
def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """
    from SuperGlue/Magic Leap
    Perform Sinkhorn Normalization in Log-space for stability
    """
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


################################################################################
def log_optimal_transport(scores, alpha, iters: int):
    """
    from SuperGlue/Magic Leap
    Perform Differentiable Optimal Transport in Log-space for stability
    """
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


################################################################################
def get_layers_geometric(start: int, stop: int, step: int) -> list[int]:
    """
    get an approximately geometric series from <start> to <stop> in <step> number of steps.
    e.g. start = 512, step = 4:
                    | stop = 32 | stop = 64
        layer id    |     #     |     #
            0       |    512    |    512
            1       |    256    |    304
            2       |    128    |    181
            3       |     64    |    108
            4       |     32    |     64

    :param start:
    :param stop:
    :param step:
    :return:
    """

    # get direction
    if start > stop:
        l_0, l_n = stop, start
        reverse = True
    else:
        l_0, l_n = start, stop
        reverse = False

    q = exp((log(l_n) - log(l_0)) / step)
    layers = [int(round(l_0 * q ** x)) for x in range(step + 1)]

    # make sure start and stop layers are correct
    layers[0] = l_0
    layers[-1] = l_n

    if reverse:
        layers = layers[::-1]

    return layers


################################################################################
def mlp(channels: list[int], do_bn=True):
    """
    simple Multi-layer perceptron, from SuperGlue/Magic Leap

    """
    num_layers = len(channels)
    layers = []
    for i in range(1, num_layers):
        layers.append(
            torch.nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (num_layers - 1):
            if do_bn:
                layers.append(torch.nn.BatchNorm1d(channels[i]))
            layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)


################################################################################
def arange_like(x, dim: int):
    """
    from SuperGlue/Magic Leap
    :param x:
    :param dim:
    :return:
    """
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


def scores_to_prediction(
        scores: torch.Tensor, threshold: float = 0.0, discard_bin: bool = False) -> dict[str, torch.Tensor]:
    # Get the matches with score above "match_threshold".
    m, n = scores.size()[1:3]
    if discard_bin:
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    else:
        # consider points matched to bin
        max0, max1 = scores.max(2), scores.max(1)
    indices0, indices1 = max0.indices, max1.indices
    mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
    mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
    if not discard_bin:
        # clean up possible dustbin assignments
        mutual0 = mutual0 & (indices0 != n - 1)
        mutual1 = mutual1 & (indices1 != m - 1)
    zero = scores.new_tensor(0)
    mscores0 = torch.where(mutual0, max0.values.exp(), zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
    valid0 = mutual0 & (mscores0 > threshold)
    if not discard_bin:
        # discard dust bin values
        valid1 = mutual1[:, :-1] & valid0.gather(1, indices1[:, :-1])
        valid0 = valid0[:, :-1]
        indices0 = indices0[:, :-1]
        indices1 = indices1[:, :-1]
        mscores0 = mscores0[:, :-1]
        mscores1 = mscores1[:, :-1]
        mutual0 = mutual0[:, :-1]
        mutual1 = mutual1[:, :-1]
    else:
        valid1 = mutual1 & valid0.gather(1, indices1)
    indices0_threshold = torch.where(valid0, indices0, indices0.new_tensor(-1))
    indices1_threshold = torch.where(valid1, indices1, indices1.new_tensor(-1))
    indices0 = torch.where(mutual0, indices0, indices0.new_tensor(-1))
    indices1 = torch.where(mutual1, indices1, indices1.new_tensor(-1))

    return {
        'matches0': indices0_threshold,  # use -1 for invalid match
        'matches1': indices1_threshold,  # use -1 for invalid match
        'matching_scores0': mscores0,
        'matching_scores1': mscores1,
        'matches0_all': indices0,
        'matches1_all': indices1,
    }


def weighted_transform(
        source: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
        eps: torch.float64 = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    calculate a rigid transformation mapping source points to target points in a least squares sense
    uses weighted SVD in the process
    :param source: BxNxD points
    :param target: BxNxD points
    :param weight: BxN weights
    :param eps: for numerical stability in divisions with potentially very small fractions
    :returns R: BxDxD rotation matrix
    :returns t: BxD translation vectors
    """
    # normalized weights. weights should be in the range [0, 1] (not checked)
    # this should not harm, but makes mean calculations more simple
    weight = weight / (torch.sum(weight, dim=1, keepdim=True) + eps)
    if weight.ndim == source.ndim - 1:
        weight = weight.unsqueeze(2)

    source_mean = torch.sum(source * weight, dim=1, keepdim=True)
    target_mean = torch.sum(target * weight, dim=1, keepdim=True)
    source_centered = source - source_mean
    target_centered = target - target_mean

    matmul_state = torch.backends.cuda.matmul.allow_tf32
    if source.device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
    covariance = torch.matmul(
        source_centered.transpose(1, 2),
        target_centered * weight,
        ).double()

    # 3x3 (DxD) matrix. do this on CPU in double precision
    u, _, v = covariance.svd()
    r_reg = torch.matmul(v, u.transpose(1, 2))
    v_neg = v.clone()
    v_neg[:, -1] *= -1.
    r_neg = torch.matmul(v_neg, u.transpose(1, 2))
    r_mat = torch.where((torch.det(r_reg) > 0)[:, None, None], r_reg, r_neg)

    if not torch.all(torch.det(r_mat) > 0.):
        raise RuntimeError("rotation matrix calculation failed, inflections encountered")

    t_vec = target_mean.double() - torch.matmul(source_mean.double(), r_mat.transpose(1, 2))

    if source.device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = matmul_state

    return r_mat, t_vec.view(-1, 3)


def estimate_transform(
        source: torch.Tensor,
        target: torch.Tensor,
        prediction: Union[dict, torch.Tensor],
        eps: torch.float64 = 1e-10,
        dtype: torch.dtype = None,
        **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    given source and target points and predicted point correspondences, estimate rigid transformations between them

    :param source: BxNx3 source point cloud
    :param target: BxMx3 target point cloud
    :param prediction: dict: predictions as returned by scores_to_prediction(), torch.Tensor: score matrix BxNxM
    :param eps: for numerical stability in divisions with potentially very small fractions
    :param dtype: datatype for calculations. input tensors will be promoted this type
    :param kwargs: for compatibility with adaptive version
    :returns: rotation matrices Bx3x3, translation vectors Bx3 with same type as source
    """
    device = source.device
    if dtype is None:
        new_type = source.dtype
        old_type = None
    else:
        new_type = dtype
        old_type = source.dtype

    source = source.to(device=device, dtype=new_type)
    target = target.to(device=device, dtype=new_type)

    if isinstance(prediction, dict):
        correspondence = prediction['matches1'].to(device=device, dtype=torch.long)
        score = prediction['matching_scores1'].to(device=device, dtype=new_type)
        correspondence_valid = correspondence >= 0
        correspondence_valid[correspondence_valid.sum(1) < 3, :] = False

        # gather does not work with -1 indices. these won't make any difference due to the weights anyway!
        correspondence = torch.where(correspondence_valid, correspondence, 0)

        zero = source.new_tensor(0., device=device, dtype=new_type)
        source = torch.gather(source, 1, correspondence.unsqueeze(2).expand((-1, -1, 3)))
        source = torch.where(correspondence_valid.unsqueeze(2), source, zero)
        target = torch.where(correspondence_valid.unsqueeze(2), target, zero)
        score = torch.where(correspondence_valid, score, zero)
    else:
        score = prediction[:, :source.shape[1], :target.shape[1]].to(device=device, dtype=new_type)
        # this transformation expects score to be (approximately) a permutation matrix
        # all elements are 0 <= s <= 1, and the sum already sums up to one (sinkhorn)
        # it assumes, that the sum across a column signifies reduced reliability of the result
        weight = score.sum(1, keepdim=True)
        score = score / (weight + eps)
        source = torch.matmul(score.transpose(2, 1), source.transpose(2, 1)).contiguous()
        score = weight

    rotation, translation = weighted_transform(
        source,
        target,
        score,
        eps,
    )

    if old_type is not None:
        rotation = rotation.to(old_type)
        translation = translation.to(old_type)

    return rotation, translation
