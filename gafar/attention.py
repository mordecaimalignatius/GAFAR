"""
3D attention/transformer modules for point clouds

Ludwig Mohr
ludwig.mohr@icg.tugraz.at

"""
import torch
from torch import nn
from abc import ABC, abstractmethod
from copy import copy

from utils.misc import merge_dict
from .misc import mlp
from .norm import AttentionBatchNorm


################################################################################
class Attention(nn.Module, ABC):
    def __init__(self, dim: int, dropout: float = 0.0):
        """
        for compatibility between attention layer implementations

        """
        super(Attention, self).__init__()
        self._dim = dim
        self._dropout = nn.Dropout(p=dropout) if 0.0 < dropout < 1.0 else None

    @abstractmethod
    def forward(self, query: torch.tensor, key: torch.tensor, value: torch.tensor):
        """

        :param query: B x D_attn x N_head x M
        :param key: B x D_attn x N_head x M
        :param value: B x D_attn x N_head x M
        :return:
        """
        pass


class SoftmaxAttention(Attention):
    def __init__(self, dim: int, dropout: float = 0.0):
        """
        regular softmax attention

        :param dim: feature dimension to attend over
        """
        super(SoftmaxAttention, self).__init__(dim, dropout=dropout)
        self._norm = 1. / (float(dim) ** 0.5)

    def forward(self, query: torch.tensor, key: torch.tensor, value: torch.tensor):
        """
        regular softmax attention over all M data points

        :param query: B x D_attn x N_head x N
        :param key: B x D_attn x N_head x M
        :param value: B x D_attn x N_head x M
        :return:
        """
        prob = torch.einsum('bdhn,bdhm->bhnm', query, key) * self._norm
        prob = nn.functional.softmax(prob, dim=-1)
        if self._dropout is not None:
            prob = self._dropout(prob)
        return torch.einsum('bhnm,bdhm->bdhn', prob, value)


################################################################################
class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            num_heads: int,
            dim_model: int,
            dropout: float = 0.0,
    ):
        """
        implementation of a single graph multi-head attention layer for fully connected attention

        :param num_heads: number of attention heads
        :param dim_model: dimension of the model
        :param dropout: dropout probability in attention
        """
        super(MultiHeadAttention, self).__init__()
        assert dim_model % num_heads == 0, 'number of heads must be an integer divisor of feature dimension'
        self._dim = dim_model // num_heads
        self._num_heads = num_heads

        self._merge = nn.Conv1d(dim_model, dim_model, kernel_size=1)
        self._proj = nn.ModuleList([nn.Conv1d(dim_model, dim_model, kernel_size=1) for _ in range(3)])
        self._attn = SoftmaxAttention(self._dim, dropout=dropout)

    def forward(self, query: torch.tensor, key: torch.tensor, value: torch.tensor):
        """
        Multi Head Attention Layer performing global attention

        :param query: BxDxN query tensor
        :param key: BxDxM key tensor
        :param value: BxDxM value tensor
        :param kwargs: compatibility for local attention/neighbourhood index
        :return:
        """
        batch_size = query.size(0)

        # linear projection, reshape for number of heads
        query = self._proj[0](query).view(batch_size, self._dim, self._num_heads, -1)   # B x D_attn x H x N
        key = self._proj[1](key).view(batch_size, self._dim, self._num_heads, -1)       # B x D_attn x H x M
        value = self._proj[2](value).view(batch_size, self._dim, self._num_heads, -1)   # B x D_attn x H x M

        # aggregate keys and values according to neighbourhood info
        attn = self._attn(query, key, value)

        return self._merge(attn.contiguous().view(batch_size, self._dim * self._num_heads, -1))


################################################################################
class GraphDualAttentionPropagation(nn.Module, ABC):
    """
    attentional propagation layer base implementation for dual graph self/cross attention

    """
    _default = {
        'feature_dimension': 32,    # dimension of per point features [int]
        'heads': 1,                 # number of heads in multi-head-attention [int]
        'merge': 'add',             # merging method to merge residual connection ['add', 'cat']
        'norm': False,              # normalization layer after residual connection [bool, dict]
        'dropout': 0.0,             # dropout probability of attention update [float]
        'attention_dropout': 0.0,   # dropout probability in attention module/inner product (softmax only) [float]
    }

    # default norm
    _default_norm = {
        'dims': 1,
        'eps': 1e-6,
    }

    def __init__(self, config: dict):
        super(GraphDualAttentionPropagation, self).__init__()
        self._config = merge_dict(self._default, config)

        mha = MultiHeadAttention(
            self._config['heads'],
            self._config['feature_dimension'],
            dropout=self._config['attention_dropout']
        )
        if 0.0 < self._config['dropout'] < 1.0:
            self._attn = nn.Sequential(mha, nn.Dropout(p=self._config['dropout']))
        else:
            self._attn = mha

        dim = self._config['feature_dimension']
        if self._config['norm']:
            if isinstance(self._config['norm'], dict):
                norm_config = self._config['norm']
                if 'type' not in norm_config:
                    norm_config['type'] = self._default_norm['type']
            else:
                # default norm
                norm_config = self._default_norm
            norm_config['feature_dimension'] = dim
            norm = AttentionBatchNorm(self._config['norm'])
            embedding = mlp([dim*2, dim*2, dim])
            torch.nn.init.constant_(embedding[-1].bias, 0.0)
            self._embedding = nn.Sequential(embedding, norm)
            self._config['norm'] = norm.config

        else:
            self._embedding = mlp([dim*2, dim*2, dim])
            torch.nn.init.constant_(self._embedding[-1].bias, 0.0)

        if self._config['merge'] == 'cat':
            self._merge_mlp = mlp([dim * 2, dim * 2, dim])
            torch.nn.init.constant_(self._merge_mlp[-1].bias, 0.0)
            self._merge = self._cat

        else:
            self._merge_mlp = None
            self._merge = self._add

    @staticmethod
    def _add(features, message):
        """
        regular residual connection

        :param features:
        :param message:
        :return:
        """
        return features + message

    def _cat(self, features, message):
        """
        learnable residual connection

        :param features:
        :param message:
        :return:
        """
        return self._merge_mlp(torch.cat([features, message], dim=1))

    def forward(self, features):
        pass

    @property
    def config(self) -> dict:
        return copy(self._config)


class GraphSelfAttentionPropagation(GraphDualAttentionPropagation):
    def __init__(self, config: dict):
        """
        attentional propagation layer for dual single source/self attention

        """
        super(GraphSelfAttentionPropagation, self).__init__(config)
        self._config['type'] = 'self'

    def forward(self, features):
        message = self._attn(features[0], features[0], features[0])
        features_0 = self._embedding(torch.cat([features[0], message], dim=1))
        message = self._attn(features[1], features[1], features[1])
        features_1 = self._embedding(torch.cat([features[1], message], dim=1))
        return self._merge(features[0], features_0), self._merge(features[1], features_1)


class GraphCrossAttentionPropagation(GraphDualAttentionPropagation):
    def __init__(self, config: dict):
        """
        attentional propagation layer for cross attention

        """
        super(GraphCrossAttentionPropagation, self).__init__(config)
        self._config['type'] = 'cross'

    def forward(self, features):
        message_0 = self._attn(features[0], features[1], features[1])
        message_1 = self._attn(features[1], features[0], features[0])

        features_0 = self._embedding(torch.cat([features[0], message_0], dim=1))
        features_1 = self._embedding(torch.cat([features[1], message_1], dim=1))
        return self._merge(features[0], features_0), self._merge(features[1], features_1)


################################################################################
class GraphCrossAttentionNetwork(nn.Module):
    """
    graph attention network for two point sets with inter and cross point set attention connections

    """
    _default = {
        'feature_dimension': 32,            # feature dimension
        'layers': ['self', 'cross'] * 9,    # layer definition, any of [self, cross, norm, <norm type>] or config dicts
        'heads': 1,                         # number of multi-head-attention heads
        'merge': 'add',                     # add, cat (plus mlp)
        # 'norm": {type: 'group'},          # optional default configuration for normalization layers
    }

    # default norm
    _default_norm = {
        'type': 'batch',        # batch, group, layer
        'dims': 1,              # 1D batch norm
        'eps': 1e-6,
    }

    def __init__(self, config: dict):
        super(GraphCrossAttentionNetwork, self).__init__()
        self._config = merge_dict(self._default, config)

        modules = nn.ModuleList()
        modules_config = []
        attention_propagation_default = {
            'feature_dimension': self._config['feature_dimension'],
            'heads': self._config['heads'],
            'merge': self._config['merge'],
        }
        for layer in self._config['layers']:
            # if str, default initialization AttentionPropagation [cross, self], Norm [batch, instance, group]
            if isinstance(layer, str):
                if layer in ['batch', 'norm']:
                    # normalization layer from register
                    layer_config = self._config['norm'] if 'norm' in self._config else self._default_norm
                    layer_config['feature_dimension'] = self._config['feature_dimension']
                    modules.append(AttentionBatchNorm(layer_config))

                elif layer == 'cross':
                    # cross attention propagation layer
                    modules.append(GraphCrossAttentionPropagation(attention_propagation_default))

                elif layer == 'self':
                    # self attention propagation layer
                    modules.append(GraphSelfAttentionPropagation(attention_propagation_default))

                else:
                    raise RuntimeError(f'unknown graph attention module: \"{layer}\"')

            elif isinstance(layer, dict) and 'type' in layer:
                # configuration dictionary for layer
                layer['feature_dimension'] = self._config['feature_dimension']
                if layer['type'] == 'batch':
                    modules.append(AttentionBatchNorm(layer))

                elif layer['type'] == 'cross':
                    layer_config = merge_dict(attention_propagation_default, layer)
                    modules.append(GraphCrossAttentionPropagation(layer_config))

                elif layer['type'] == 'self':
                    layer_config = merge_dict(attention_propagation_default, layer)
                    modules.append(GraphSelfAttentionPropagation(layer_config))

                else:
                    raise RuntimeError(f'unknown layer type {layer["type"]}')

            else:
                raise RuntimeError(f'unknown layer definition of type {type(layer)}')

            modules_config.append(modules[-1].config)

        self._config['layers'] = modules_config
        self._model = nn.Sequential(*modules)

    def forward(self, features_0, features_1):
        return self._model((features_0, features_1))

    @property
    def config(self) -> dict:
        return self._config
