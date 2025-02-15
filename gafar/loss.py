"""
losses for training neural networks for point cloud registration

Ludwig Mohr
ludwig.mohr@icg.tugraz.at

"""
import torch
import torch.nn.functional


from .misc import weighted_transform


########################################################################################################################
class Loss(torch.nn.Module):
    def __init__(
            self,
            use_score: bool = True,
            use_transformation: bool = False,
            lambda_s: float = 1.,
            lambda_t0: float = 1.,
            lambda_t1: float = 1.,
            score_loss_type: str = 'lightglue',
            eps: float = torch.finfo(torch.float32).eps,
            nll_balance: float = 0.5,
            matchability_by_distance: bool = False,
            inlier: float = 0.05,
    ):
        """

        :param use_score:
        :param use_transformation:
        :param lambda_s:
        :param lambda_t0:
        :param lambda_t1:
        :param eps:
        :param nll_balance: balancing between negative and positive influence for NLLLoss for light glue type matching
        :param matchability_by_distance: if True, use inlier distance to determine matchability
        :param inlier: point distance to be considered a possible positive (match, inlier) in contrastive loss
        """
        super(Loss, self).__init__()

        if not (use_score or use_transformation):
            raise RuntimeError('must select at least one loss')

        assert 0.0 < nll_balance < 1.0, f'balancing weight between matched and unmatched must be in (0.0, 1.0), ' \
                                        f'is {nll_balance:.f}'

        if score_loss_type.lower() == 'rgm':
            self.score = self.bce_permutation_loss
        elif score_loss_type.lower() == 'lightglue':
            self.score = self.bce_matchable
        else:
            raise RuntimeError(f'unknown score loss calculation: \"{score_loss_type}\"')

        self._nll_balance = nll_balance
        self._inlier = inlier
        self._matchability_by_distance = matchability_by_distance

        self._lambda_score = lambda_s
        self._lambda_t0 = lambda_t0
        self._lambda_t1 = lambda_t1
        self._eps = eps

        self._use_score_loss = use_score
        self._use_transformation_loss = use_transformation
        self._calculate_score_loss = lambda_s > 0.
        self._calculate_transformation_loss = lambda_t0 > 0. or lambda_t1 > 0.

        self._float_min = -1. + self._eps
        self._float_max = 1. - self._eps

    def forward(
            self,
            data: dict[str, torch.Tensor],
    ):
        """
        data needs to have fields:
        score           matching score matrix, either BxMxN (no dustbins) or BxM+1xN+1 (with dustbins)

        depending on specific loss to be calculated one or more of the following fields must be present:
        points_0        source points, BxMx3
        points_1        reference points, BxNx3
        correspondence  (ground truth), BxN
        rotation        (ground truth), Bx3x3
        translation     (ground truth), Bx3

        for network with masks, the following two fields may be present:
        valid0          valid points in source, BxM torch.bool
        valid1          valid points in reference, BxN torch.bool
        """
        d_type = data['score'].dtype
        device = data['score'].device

        loss = torch.tensor(0., device=device, dtype=d_type)
        score_loss = torch.tensor(torch.nan)
        rotation_loss = torch.tensor(torch.nan)
        translation_loss = torch.tensor(torch.nan)

        data = self.point_projection(data)
        data = self.transform(data)

        if self._calculate_score_loss:
            # calculate score based loss
            score_loss = self.score(data)

            if self._use_score_loss:
                loss += self._lambda_score * score_loss

            score_loss = self._lambda_score * score_loss.detach().cpu()

        if self._calculate_transformation_loss:
            rotation_loss, translation_loss = self.transformation_loss(data)

            if self._use_transformation_loss:
                loss += self._lambda_t0 * rotation_loss.sum() + self._lambda_t1 * translation_loss.sum()

            rotation_loss = self._lambda_t0 * rotation_loss.detach().cpu()
            translation_loss = self._lambda_t1 * translation_loss.detach().cpu()

        return {
            'loss': loss,
            's_loss': score_loss,
            'r_loss': rotation_loss,
            't_loss': translation_loss,
            'rotation': data['rotation_estimate'].detach(),
            'translation': data['translation_estimate'].detach(),
        }

    def point_projection(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """ needs fields score and points_0. returns the estimated ('nn') point projection into reference """
        # calculate weights from score
        mask = False
        valid0, valid1 = None, None
        if 'valid0' in data:
            mask = True
            valid0 = data['valid0']
            valid1 = data['valid1']

        score = data['score']

        # estimate weighted transform from points_0 to points_1 according to score matrix
        if score.shape[1] == data['points_0'].shape[1] + 1:
            weights = score[:, :-1, :-1]
            if mask:
                weights[torch.logical_not(valid0).unsqueeze(2).expand(-1, -1, weights.shape[2])] = -torch.inf
            weights = weights.exp()
            weights_norm = weights.sum(dim=1, keepdim=True)
            weights = weights / (weights_norm + self._eps)
        else:
            weights = score
            weights_norm = weights.new_ones((weights.shape[0], 1, weights.shape[2]))

        points_0_t = torch.matmul(weights.transpose(1, 2), data['points_0'])

        if mask:
            weights_norm[torch.logical_not(valid1).unsqueeze(1)] = 0.0

        data['points_0_1'] = points_0_t
        data['weights'] = weights_norm
        return data

    def transform(self, data: dict[str, torch.Tensor]):
        """ estimate weighted transform from points_0_t to points_1 """
        rot, trans = weighted_transform(
            data['points_0_1'], data['points_1'], data['weights'].transpose(1, 2), eps=self._eps)

        data['rotation_estimate'] = rot
        data['translation_estimate'] = trans
        return data

    @staticmethod
    def bce_permutation_loss(data: dict[str, torch.Tensor]) -> torch.Tensor:
        """ RGM like full BCE loss, but only on permutation portion of the score matrix (i.e. ignoring dust bins) """
        mask = False
        not_valid = None
        if 'valid0' in data:
            mask = True
            not_valid = torch.logical_and(torch.logical_not(data['valid0']).unsqueeze(2),
                                          torch.logical_not(data['valid1']).unsqueeze(1))
        score = data['score']
        correspondence = data['correspondence']

        batch_size = score.shape[0]
        truth = torch.zeros_like(score[:, :-1, :-1])
        valid = correspondence > -1
        batch_idx = torch.arange(batch_size, device=score.device).view(-1, 1).expand(-1, correspondence.shape[1])
        reference_idx = torch.arange(score.shape[2]-1, device=score.device).view(1, -1).expand(batch_size, -1)
        truth[batch_idx[valid].flatten(), correspondence[valid].flatten(), reference_idx[valid].flatten()] = 1.0

        score = score[:, :-1, :-1]
        if mask:
            score[not_valid] = -100

        return torch.nn.functional.binary_cross_entropy(score.exp(), truth, reduction='sum')

    def bce_matchable(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        BCE as described in LightGlue paper:
        - 1/|M| sum( log(P_m) ) - 1/|A| sum( log(1-s_A) ) - 1/|B| sum( log(1-s_B) )

        """
        mask = False
        not_valid0, not_valid1 = None, None
        score = data['score']
        correspondence = data['correspondence']
        if 'valid0' in data:
            mask = True
            # remove masked values from correspondence
            not_valid0 = torch.logical_not(data['valid0'])
            not_valid1 = torch.logical_not(data['valid1'])
            # set masked points in reference to unmatched
            correspondence[not_valid1] = -1
            # location of matched points in correspondence vector
            matches = correspondence > -1
            # boolean index of points matched to masked source points
            index = correspondence.clone()
            index[torch.logical_not(matches)] = 0
            matched_not_valid = torch.gather(not_valid0, 1, index)
            # set points in reference matched to masked points in source to unmatched
            correspondence[torch.logical_and(matched_not_valid, matches)] = -1

        batch_size, num_m, num_n = score.shape
        num_m_v, num_n_v = num_m - 1, num_n - 1
        if mask:
            num_m_v = num_m - not_valid0.sum(1) - 1
            num_n_v = num_n - not_valid1.sum(1) - 1

        matches = correspondence > -1
        num_matches = matches.sum(1)

        weight_match = 1.0 / num_matches

        # matched points
        batch_idx_match, ref_idx_match = torch.nonzero(matches, as_tuple=True)
        src_idx_match = correspondence[batch_idx_match, ref_idx_match]
        loss_match = torch.mul(
            score[batch_idx_match, src_idx_match, ref_idx_match],
            weight_match[batch_idx_match]).sum()

        if self._matchability_by_distance:
            if 'distance' in data:
                distance = data['distance']
            else:
                with torch.no_grad():
                    distance = self.distance(
                        data['points_0'] @ data['rotation'].to(data['points_0']).transpose(1, 2) +
                        data['translation'].to(data['points_0']).unsqueeze(1),
                        data['points_1'])
                data['distance'] = distance

            too_far = distance >= self._inlier
            unmatched_src = too_far.sum(2) == num_n - 1
            unmatched_ref = too_far.sum(1) == num_m - 1

        else:
            # unmatched points in source
            unmatched_src = torch.ones((batch_size, num_m-1), device=matches.device, dtype=torch.bool)
            unmatched_src[batch_idx_match, src_idx_match] = False
            # unmatched points in reference
            unmatched_ref = correspondence == -1

        # mask matchability indices
        if mask:
            unmatched_src[not_valid0] = False
            unmatched_ref[not_valid1] = False

        # calculate matchability weights
        if self._matchability_by_distance:
            weight_m = 0.5 / unmatched_src.sum(1).clamp(min=1.0)
            weight_n = 0.5 / unmatched_ref.sum(1).clamp(min=1.0)
        else:
            weight_m = 0.5 / (num_m_v - num_matches).clamp(min=1.0)
            weight_n = 0.5 / (num_n_v - num_matches).clamp(min=1.0)

        batch_idx_bin_src, src_idx_bin = torch.nonzero(unmatched_src, as_tuple=True)
        loss_bin_ref = torch.mul(score[batch_idx_bin_src, src_idx_bin, -1], weight_m[batch_idx_bin_src]).sum()

        batch_idx_bin_ref, ref_idx_bin = torch.nonzero(unmatched_ref, as_tuple=True)
        loss_bin_src = torch.mul(score[batch_idx_bin_ref, -1, ref_idx_bin], weight_n[batch_idx_bin_ref]).sum()

        return - self._nll_balance * loss_match - ((1.0 - self._nll_balance) * (loss_bin_ref + loss_bin_src))

    def transformation_loss(self, data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        rotation_residual = torch.matmul(data['rotation'].transpose(1, 2), data['rotation_estimate'])
        rotation_error = torch.arccos(
            torch.clamp((torch.einsum('jii->j', rotation_residual) - 1.) / 2.,
                        min=self._float_min, max=self._float_max))

        translation_error = torch.linalg.norm(data['translation'] - data['translation_estimate'], dim=1)

        return rotation_error, translation_error

    @staticmethod
    def distance(points_0: torch.Tensor, points_1: torch.Tensor) -> torch.Tensor:
        """ mutual squared point distance """
        # linear algebra greedy nearest neighbours
        inner = 2 * torch.matmul(points_0, points_1.transpose(1, 2))  # BxMxD * BxDxN -> BxMxN
        source_2 = torch.einsum('bnd,bnd->bn', points_0, points_0).unsqueeze(2)     # BxMx1
        target_2 = torch.einsum('bnd,bnd->bn', points_1, points_1).unsqueeze(1)     # Bx1xN
        distance = source_2 - inner + target_2

        return distance.clamp(min=0.0)

    def __str__(self):
        losses = []
        if self._use_score_loss:
            losses.append('score')
        if self._use_transformation_loss:
            losses.append('transform')

        losses = ' '.join(f'\"{x}\"' for x in losses)
        lambdas = f'[score: {self._lambda_score * float(self._use_score_loss):.2f}, ' \
                  f'transform: {self._lambda_t0 * float(self._use_transformation_loss):.2f}/' \
                  f'{self._lambda_t1 * float(self._use_transformation_loss):.2f}]'

        return_str = losses + '\t' + lambdas

        return return_str

    @property
    def has_score_loss(self):
        return self._calculate_score_loss

    @property
    def has_transformation_loss(self):
        return self._calculate_transformation_loss

    @property
    def is_score_loss(self):
        return self._use_score_loss

    @property
    def is_transformation_loss(self):
        return self._use_transformation_loss
