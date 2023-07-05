"""
losses for training neural networks for point cloud registration

Ludwig Mohr
ludwig.mohr@icg.tugraz.at

"""
import torch
import torch.nn.functional


from .misc import weighted_transform


################################################################################
class Loss(torch.nn.Module):
    def __init__(
            self,
            use_score: bool = True,
            use_transformation: bool = False,
            lambda_s: float = 1.,
            lambda_t0: float = 1.,
            lambda_t1: float = 1.,
            eps: float = torch.finfo(torch.float32).eps,
    ):
        """

        :param use_score:
        :param use_transformation:
        :param lambda_s:
        :param lambda_t0:
        :param lambda_t1:
        :param eps:
        """
        super(Loss, self).__init__()

        if not (use_score or use_transformation):
            raise RuntimeError('must select at least one loss')

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
            score: torch.Tensor,
            points_0: torch.Tensor,
            points_1: torch.Tensor,
            correspondence: torch.Tensor = None,
            rotation: torch.Tensor = None,
            translation: torch.Tensor = None,
    ):
        loss = torch.tensor(0., device=score.device, dtype=score.dtype)
        score_loss = torch.tensor(float('nan'))
        rotation_loss = torch.tensor(float('nan'))
        translation_loss = torch.tensor(float('nan'))

        points_0_t, weights = self.point_projection(score, points_0)
        rotation_estimate, translation_estimate = self.transform(points_0_t, points_1, weights)

        if self._calculate_score_loss:
            # calculate score based loss
            score_loss = self.score(correspondence, score)

            if self._use_score_loss:
                loss += self._lambda_score * score_loss

            score_loss = self._lambda_score * score_loss.detach().cpu()

        if self._calculate_transformation_loss:
            rotation_loss, translation_loss = self.transformation_loss(
                rotation_estimate, translation_estimate, rotation, translation)

            if self._use_transformation_loss:
                loss += self._lambda_t0 * rotation_loss.sum() + self._lambda_t1 * translation_loss.sum()

            rotation_loss = self._lambda_t0 * rotation_loss.detach().cpu()
            translation_loss = self._lambda_t1 * translation_loss.detach().cpu()

        return {
            'loss': loss,
            's_loss': score_loss,
            'r_loss': rotation_loss,
            't_loss': translation_loss,
            'rotation': rotation_estimate.detach(),
            'translation': translation_estimate.detach(),
        }

    def point_projection(self, score: torch.Tensor, points_0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # calculate weights from score
        # estimate weighted transform from points_0 to points_1 according to score matrix
        if score.shape[1] == points_0.shape[1] + 1:
            weights = score[:, :-1, :-1].exp()
            weights_norm = weights.sum(dim=1, keepdim=True)
            weights = weights / (weights_norm + self._eps)
        else:
            weights = score
            weights_norm = weights.new_ones((weights.shape[0], 1, weights.shape[2]))

        points_0_t = torch.matmul(weights.transpose(1, 2), points_0)

        return points_0_t, weights_norm

    def transform(self, points_0_t: torch.Tensor, points_1: torch.Tensor, weights_norm: torch.Tensor):
        """ estimate weighted transform from points_0_t to points_1 """
        rot, trans = weighted_transform(points_0_t, points_1, weights_norm.transpose(1, 2), eps=self._eps)

        return rot, trans

    @staticmethod
    def score(correspondence: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
        """ RGM like full BCE loss, but only on permutation portion of the score matrix (i.e. ignoring dust bins) """
        batch_size = score.shape[0]
        truth = torch.zeros_like(score[:, :-1, :-1])
        valid = correspondence > -1
        batch_idx = torch.arange(batch_size, device=score.device).view(-1, 1).expand(-1, correspondence.shape[1])
        reference_idx = torch.arange(score.shape[2]-1, device=score.device).view(1, -1).expand(batch_size, -1)
        truth[batch_idx[valid].flatten(), correspondence[valid].flatten(), reference_idx[valid].flatten()] = 1.0

        return torch.nn.functional.binary_cross_entropy(score[:, :-1, :-1].exp(), truth, reduction='sum')

    def transformation_loss(
            self,
            rotation_estimate: torch.Tensor,
            translation_estimate: torch.Tensor,
            rotation: torch.Tensor,
            translation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        matmul_state = torch.backends.cuda.matmul.allow_tf32
        if rotation_estimate.device.type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = False
        rotation_residual = torch.matmul(rotation.transpose(1, 2), rotation_estimate)
        rotation_error = torch.arccos(
            torch.clamp((torch.einsum('jii->j', rotation_residual) - 1.) / 2., min=self._float_min, max=self._float_max))

        translation_error = torch.linalg.norm(translation - translation_estimate, dim=1)

        if rotation_estimate.device.type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = matmul_state

        return rotation_error, translation_error

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
