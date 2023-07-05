"""
adaption of normalization layers for use with dual attention propagation layers

Ludwig Mohr
ludwig.mohr@icg.tugraz.at

"""

import torch

from abc import ABC, abstractmethod
from copy import copy
from typing import Union

from utils.misc import merge_dict


################################################################################
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
            features: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if isinstance(features, torch.Tensor):
            return self._norm(features)

        return self._norm(features[0]), self._norm(features[1])

    @abstractmethod
    def _set_norm(self):
        # instantiate proper normalization layer as self._norm
        pass


################################################################################
class AttentionBatchNorm(AttentionNorm):
    _default = {
        'type': 'batch',
        'feature_dimension': 32,
        'dims': 1,
        'eps': 1e-06,
    }

    def __init__(self, config: dict):
        super(AttentionBatchNorm, self).__init__(config)

    def _set_norm(self):
        config = self.config
        if 'type' in config:
            config.pop('type')
        features = config.pop('feature_dimension')
        dims = config.pop('dims')
        if dims == 1:
            self._norm = torch.nn.BatchNorm1d(features, **config)
        elif dims == 2:
            self._norm = torch.nn.BatchNorm2d(features, **config)
        else:
            raise RuntimeError(f'invalid number of dimensions: {dims}')


class AttentionGroupNorm(AttentionNorm):
    _default = {
        'type': 'group',
        'groups': 2,
        'feature_dimension': 32,
        'eps': 1e-06,
    }

    def __init__(self, config: dict):
        super(AttentionGroupNorm, self).__init__(config)

    def _set_norm(self):
        config = self.config
        if 'type' in config:
            config.pop('type')
        groups = config.pop('groups')
        features = config.pop('feature_dimension')
        self._norm = torch.nn.GroupNorm(groups, features, **config)


class AttentionLayerNorm(AttentionNorm):
    _default = {
        'type': 'layer',
        'feature_dimension': 32,
        'eps': 1e-06,
    }

    def __init__(self, config: dict):
        super(AttentionLayerNorm, self).__init__(config)

    def _set_norm(self):
        config = self.config
        if 'type' in config:
            config.pop('type')
        shape = config.pop('feature_dimension')
        self._norm = torch.nn.LayerNorm(shape, **config)
