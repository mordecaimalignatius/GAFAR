"""
adaption of normalization layers for use with dual attention propagation layers

Ludwig Mohr
ludwig.mohr@icg.tugraz.at

"""

import torch

from abc import ABC, abstractmethod
from copy import copy
from typing import Union, Optional

import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm

from utils.misc import merge_dict


########################################################################################################################
class AttentionNorm(ABC, torch.nn.Module):
    """ abstract template for attention norms """
    _default = {}

    def __init__(self, config: dict):
        super(AttentionNorm, self).__init__()
        self._config = merge_dict(self._default, config)
        self._norm = None
        self._set_norm()

    @property
    def config(self) -> dict:
        return copy(self._config)

    def forward(
            self,
            features: Union[torch.Tensor, tuple[torch.Tensor, ...]],
    ) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        if isinstance(features, torch.Tensor):
            return self._norm(features)

        if len(features) == 2:
            if self.config['masked']:
                # return masked norm and mask
                b, _, m = features[0].shape
                return self._norm(features[0], features[1].view(b, 1, m)), features[1]

            return self._norm(features[0]), self._norm(features[1])

        b, _, m = features[0].shape
        _, _, n = features[2].shape
        return self._norm(features[0], features[1].view(b, 1, m)), features[1], \
               self._norm(features[2], features[3].view(b, 1, n)), features[3]

    @abstractmethod
    def _set_norm(self):
        # instantiate proper normalization layer as self._norm
        pass


########################################################################################################################
class AttentionBatchNorm(AttentionNorm):
    _default = {
        'type': 'batch',
        'feature_dimension': 32,
        'dims': 1,
        'eps': 1e-06,
        'masked': False,
    }

    def __init__(self, config: dict):
        super(AttentionBatchNorm, self).__init__(config)

    def _set_norm(self):
        config = self.config
        if 'type' in config:
            config.pop('type')
        features = config.pop('feature_dimension')
        dims = config.pop('dims')
        masked = config.pop('masked')
        if dims == 1:
            self._norm = MaskedBatchNorm1d(features, **config) if masked else torch.nn.BatchNorm1d(features, **config)
        elif dims == 2:
            self._norm = MaskedBatchNorm2d(features, **config) if masked else torch.nn.BatchNorm2d(features, **config)
        else:
            raise RuntimeError(f'invalid number of dimensions: {dims}')


class AttentionGroupNorm(AttentionNorm):
    _default = {
        'type': 'group',
        'groups': 2,
        'feature_dimension': 32,
        'eps': 1e-06,
        'masked': False,
    }

    def __init__(self, config: dict):
        super(AttentionGroupNorm, self).__init__(config)

    def _set_norm(self):
        config = self.config
        if 'type' in config:
            config.pop('type')
        groups = config.pop('groups')
        features = config.pop('feature_dimension')
        masked = config.pop('masked')
        if masked:
            raise NotImplementedError('Masked AttentionGroupNorm not implemented')
        self._norm = torch.nn.GroupNorm(groups, features, **config)


class AttentionLayerNorm(AttentionNorm):
    _default = {
        'type': 'layer',
        'feature_dimension': 32,
        'eps': 1e-06,
        'masked': False,
    }

    def __init__(self, config: dict):
        super(AttentionLayerNorm, self).__init__(config)

    def _set_norm(self):
        config = self.config
        if 'type' in config:
            config.pop('type')
        shape = config.pop('feature_dimension')
        masked = config.pop('masked')
        if masked:
            raise NotImplementedError('Masked AttentionLayerNorm not implemented')
        self._norm = torch.nn.LayerNorm(shape, **config)


########################################################################################################################
# from https://gist.github.com/ilya16
def masked_batch_norm(input: Tensor, mask: Tensor, weight: Optional[Tensor], bias: Optional[Tensor],
                      running_mean: Optional[Tensor], running_var: Optional[Tensor], training: bool,
                      momentum: float, eps: float = 1e-5) -> Tensor:
    r"""Applies Masked Batch Normalization for each channel in each data sample in a batch.

    See :class:`~MaskedBatchNorm1d`, :class:`~MaskedBatchNorm2d`, :class:`~MaskedBatchNorm3d` for details.
    """
    if not training and (running_mean is None or running_var is None):
        raise ValueError('Expected running_mean and running_var to be not None when training=False')

    num_dims = len(input.shape[2:])
    _dims = (0,) + tuple(range(-num_dims, 0))
    _slice = (None, ...) + (None,) * num_dims

    if training:
        num_elements = mask.sum(_dims)
        mean = (input * mask).sum(_dims) / num_elements  # (C,)
        var = (((input - mean[_slice]) * mask) ** 2).sum(_dims) / num_elements  # (C,)

        if running_mean is not None:
            running_mean.copy_(running_mean * (1 - momentum) + momentum * mean.detach())
        if running_var is not None:
            running_var.copy_(running_var * (1 - momentum) + momentum * var.detach())
    else:
        mean, var = running_mean, running_var

    out = (input - mean[_slice]) / torch.sqrt(var[_slice] + eps)  # (N, C, ...)

    if weight is not None and bias is not None:
        out = out * weight[_slice] + bias[_slice]

    return out


class _MaskedBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_MaskedBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input: Tensor, mask: Tensor = None) -> Tensor:
        self._check_input_dim(input)
        if mask is not None:
            self._check_input_dim(mask)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        if mask is None:
            return F.batch_norm(
                input,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.running_mean if not self.training or self.track_running_stats else None,
                self.running_var if not self.training or self.track_running_stats else None,
                self.weight, self.bias, bn_training, exponential_average_factor, self.eps
            )
        else:
            return masked_batch_norm(
                input, mask, self.weight, self.bias,
                self.running_mean if not self.training or self.track_running_stats else None,
                self.running_var if not self.training or self.track_running_stats else None,
                bn_training, exponential_average_factor, self.eps
            )


class MaskedBatchNorm1d(torch.nn.BatchNorm1d, _MaskedBatchNorm):
    r"""Applies Batch Normalization over a masked 3D input
    (a mini-batch of 1D inputs with additional channel dimension)..

    See documentation of :class:`~torch.nn.BatchNorm1d` for details.

    Shape:
        - Input: :math:`(N, C, L)`
        - Mask: :math:`(N, 1, L)`
        - Output: :math:`(N, C, L)` (same shape as input)
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True) -> None:
        super(MaskedBatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)


class MaskedBatchNorm2d(torch.nn.BatchNorm2d, _MaskedBatchNorm):
    r"""Applies Batch Normalization over a masked 4D input
    (a mini-batch of 2D inputs with additional channel dimension)..

    See documentation of :class:`~torch.nn.BatchNorm2d` for details.

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Mask: :math:`(N, 1, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True) -> None:
        super(MaskedBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)


class MaskedBatchNorm3d(torch.nn.BatchNorm3d, _MaskedBatchNorm):
    r"""Applies Batch Normalization over a masked 5D input
    (a mini-batch of 3D inputs with additional channel dimension).

    See documentation of :class:`~torch.nn.BatchNorm3d` for details.

    Shape:
        - Input: :math:`(N, C, D, H, W)`
        - Mask: :math:`(N, 1, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True) -> None:
        super(MaskedBatchNorm3d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
