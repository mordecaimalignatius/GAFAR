import torch

from gafar.misc import weighted_transform_single


########################################################################################################################
def irls_sigmoid_single(source: torch.Tensor, reference: torch.Tensor,
                        rotation: torch.Tensor, translation: torch.Tensor,
                        distance: float, cutoff: float = 0.33, sigmoid_swing: float = 0.95, iteration: int = 10,
                        mode: str = 'hard',
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    run IRLS with re-weighting scheme using sigmoid re-weighting based on maximum allowed point distance
    for a single full-scale point cloud

    source: Mx3 points source[i] is considered as correspondence of reference[i]
    reference: Mx3 points
    rotation: 3x3 initial rotation
    translation: 3x initial translation
    distance: inlier distance
    cutoff: relative distance where the weight will reach sigmoid_swing % of the range
    mode: ['soft'], ['hard'] if soft, do sigmoid distance weighting. if hard, cut off at distance.
    iteration: number of re-weighting estimation iterations

    returns:
        rotation matrix Bx3x3
        translation vector Bx3
        weights BxN

    """
    assert reference.ndim == 2 and reference.shape[1] == 3, "reference must be Nx3"
    assert source.ndim == 2 and source.shape[1] == 3, "source must be Mx3"
    assert mode in ['soft', 'hard'], "IRLS re-weighting mode must be 'soft' or 'hard'"

    # failed registration attempt
    if source.shape[0] < 3:
        return rotation, translation, source.new_zeros((source.shape[0]),)

    gamma = - 1.0/(cutoff * distance) * torch.log(torch.tensor(2.0/(1. + sigmoid_swing) - 1.0)).to(source)

    src_points = source @ rotation.transpose(0, 1) + translation.unsqueeze(0)

    rotation_ = torch.clone(rotation)
    translation_ = torch.clone(translation)

    weights = torch.ones((source.shape[0]), dtype=torch.bool, device=source.device)

    for idx in range(iteration):
        # point distance
        point_distance = torch.linalg.norm(src_points - reference, dim=1)

        # update weights and valid points
        if mode == 'soft':
            weights = torch.sigmoid(-(point_distance - distance) * gamma)
        else:
            weights = torch.lt(point_distance, distance).to(point_distance)

        # "at least three points right at the cutoff distance"
        if weights.sum() < (1.5 if mode == 'soft' else 3):
            print(f'IRLS stopping at iteration {idx}, {source.shape[0]} input points')
            break

        # backup and re-estimate transformation
        rotation_, translation_ = torch.clone(rotation), torch.clone(translation_)
        rotation, translation = weighted_transform_single(source, reference, weights)

        # apply transformation
        src_points = source @ rotation.transpose(0, 1) + translation.unsqueeze(0)

    # point distance
    point_distance = torch.linalg.norm(src_points - reference, dim=1)
    num_in_range = torch.lt(point_distance, distance).sum()
    if num_in_range < 3.0:
        return rotation_, translation_, weights

    return rotation, translation, weights
