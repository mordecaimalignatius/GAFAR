"""
3D attention/transformer modules for point clouds

Ludwig Mohr
ludwig.mohr@icg.tugraz.at

"""
import torch
from torch import nn
from abc import ABC, abstractmethod
from copy import copy
from typing import Callable

from utils.misc import merge_dict
from .feature import MLP
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


class MaskedSoftmaxAttention(SoftmaxAttention):
    def forward(self, query: torch.tensor, key: torch.tensor, value: torch.tensor,
                mask: tuple[torch.Tensor, torch.Tensor] = None):
        """
        regular softmax attention over all M data points

        :param query: B x D_attn x N_head x N
        :param key: B x D_attn x N_head x M
        :param value: B x D_attn x N_head x M
        :param mask: tuple of mask tensors for query and key/value [B x 1 x N, B x 1 x M]
        :return:
        """
        if mask is None:
            raise RuntimeError('no mask supplied for MaskedSoftmaxAttention. '
                               'Use regular SoftmaxAttention instead or supply mask')
        prob = (torch.einsum('bdhn,bdhm->bhnm', query, key) * self._norm)
        # mask values for softmax computation
        prob[torch.logical_not(mask[1])[:, :, None, :].expand(prob.shape)] = -torch.inf
        # prob /= prob.sum(-1, keepdim=True)
        prob = nn.functional.softmax(prob, dim=-1)  # B x N_head x N x M

        if self._dropout is not None:
            prob = self._dropout(prob)

        attention = torch.einsum('bhnm,bdhm->bdhn', prob, value)
        return attention * mask[0][:, :, None, :]


################################################################################
class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            num_heads: int,
            dim_model: int,
            dropout: float = 0.0,
            masked: bool = False,
    ):
        """
        implementation of a single graph multi-head attention layer for fully connected attention

        :param num_heads: number of attention heads
        :param dim_model: dimension of the model
        :param dropout: dropout probability in attention
        :param masked: use masked attention layers
        """
        super(MultiHeadAttention, self).__init__()
        assert dim_model % num_heads == 0, 'number of heads must be an integer divisor of feature dimension'
        self._dim = dim_model // num_heads
        self._num_heads = num_heads

        self._merge = nn.Conv1d(dim_model, dim_model, kernel_size=1)
        self._proj = nn.ModuleList([nn.Conv1d(dim_model, dim_model, kernel_size=1) for _ in range(3)])
        self._attn = MaskedSoftmaxAttention(self._dim, dropout=dropout) if masked \
            else SoftmaxAttention(self._dim, dropout=dropout)

    def forward(self, query: torch.tensor, key: torch.tensor, value: torch.tensor, **kwargs):
        """
        Multi Head Attention Layer performing global attention

        :param query: BxDxN query tensor
        :param key: BxDxM key tensor
        :param value: BxDxM value tensor
        :param kwargs: compatibility for local attention/neighbourhood index
        :return:
        """
        batch_size = query.size(0)

        mask = None
        if 'mask' in kwargs and kwargs['mask'] is not None:
            if isinstance(kwargs['mask'], torch.Tensor):
                mask = (kwargs['mask'], kwargs['mask'])
            else:
                mask = kwargs['mask']   # tuple[torch.Tensor, torch.Tensor]

        # linear projection, reshape for number of heads
        query = self._proj[0](query).view(batch_size, self._dim, self._num_heads, -1)   # B x D_attn x H x N
        key = self._proj[1](key).view(batch_size, self._dim, self._num_heads, -1)       # B x D_attn x H x M
        value = self._proj[2](value).view(batch_size, self._dim, self._num_heads, -1)   # B x D_attn x H x M

        # apply rotational encoding
        if 'encoding' in kwargs and kwargs['encoding'] is not None:
            encoding = kwargs['encoding']
            query = query * encoding[0] + self.rotate_half(query) * encoding[1]
            key = key * encoding[0] + self.rotate_half(key) * encoding[1]

        # aggregate keys and values according to neighbourhood info
        if mask is not None:
            attn = self._attn(query, key, value, mask=mask)
        else:
            attn = self._attn(query, key, value)

        return self._merge(attn.contiguous().view(batch_size, self._dim * self._num_heads, -1))

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        # B x D x H x M -> B x H x M x D
        x = x.permute(0, 2, 3, 1)
        x = x.unflatten(-1, (-1, 2))
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)
        # B x H x M x D -> B x D x H x M
        return x.permute(0, 3, 1, 2).contiguous()


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
        'masked': False,            # use masked multi-head attention
    }

    # default norm
    _default_norm = {
        'dims': 1,
        'eps': 1e-6,
    }

    # standard embedding MLP configuration
    def _embedding_default(self, dim: int):
        return {
            'input_dimension': dim * 2,
            'feature_dimension': dim,
            'channels': [dim * 2, dim * 2, dim],
            'masked': self._config['masked'],
            'init': 'last',
        }

    def __init__(self, config: dict):
        super(GraphDualAttentionPropagation, self).__init__()
        self._config = merge_dict(self._default, config)

        mha = MultiHeadAttention(
            self._config['heads'],
            self._config['feature_dimension'],
            dropout=self._config['attention_dropout'],
            masked=self._config['masked'],
        )
        if 0.0 < self._config['dropout'] < 1.0:
            self._attn = nn.Sequential(mha, nn.Dropout(p=self._config['dropout']))
        else:
            self._attn = mha

        dim = self._config['feature_dimension']
        embedding_config = self._config['embedding'] if 'embedding' in self._config else {}
        embedding_default = self._embedding_default(dim)
        embedding_config = merge_dict(embedding_default, embedding_config)
        embedding = MLP(embedding_config)
        self._config['embedding'] = embedding.config

        if self._config['norm']:
            if isinstance(self._config['norm'], dict):
                norm_config = self._config['norm']
                if 'type' not in norm_config:
                    norm_config['type'] = self._default_norm['type']
            else:
                # default norm
                norm_config = self._default_norm
            if self._config['masked']:
                self._config['norm']['masked'] = self._config['masked']
            norm_config['feature_dimension'] = dim
            norm = AttentionBatchNorm(self._config['norm'])
            self._embedding = nn.Sequential(embedding, norm)
            self._config['norm'] = norm.config

        else:
            self._embedding = embedding

        if self._config['merge'] == 'cat':
            self._merge_mlp = mlp([dim * 2, dim * 2, dim])
            torch.nn.init.constant_(self._merge_mlp[-1].bias, 0.0)
            self._merge = self._cat

        else:
            self._merge_mlp = None
            self._merge = self._add

        self._encoding = None

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

    @abstractmethod
    def forward(self, features: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        pass

    @property
    def config(self) -> dict:
        return copy(self._config)

    def encoding(self, encoding: Callable[[int], torch.Tensor]):
        self._encoding = encoding


########################################################################################################################
# SuperGlue Style Transformer Block
class GraphSelfAttentionPropagation(GraphDualAttentionPropagation):
    def __init__(self, config: dict):
        """
        attentional propagation layer for dual single source/self attention

        """
        super(GraphSelfAttentionPropagation, self).__init__(config)
        self._config['type'] = 'self'

    def forward(self, features):
        encoding_0 = self._encoding(0) if self._encoding is not None else None
        encoding_1 = self._encoding(1) if self._encoding is not None else None
        if self._config['masked']:
            message = self._attn(features[0], features[0], features[0], mask=features[1], encoding=encoding_0)
            features_0 = self._embedding((torch.cat([features[0], message], dim=1), features[1]))
            message = self._attn(features[2], features[2], features[2], mask=features[3], encoding=encoding_1)
            features_1 = self._embedding((torch.cat([features[2], message], dim=1), features[3]))
            return self._merge(features[0], features_0), features[1], self._merge(features[2], features_1), features[3]

        message = self._attn(features[0], features[0], features[0], encoding=encoding_0)
        features_0 = self._embedding(torch.cat([features[0], message], dim=1))
        message = self._attn(features[1], features[1], features[1], encoding=encoding_1)
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
        if self._config['masked']:
            message_0 = self._attn(features[0], features[2], features[2], mask=(features[1], features[3]))
            message_1 = self._attn(features[2], features[0], features[0], mask=(features[3], features[1]))

            features_0 = self._embedding((torch.cat([features[0], message_0], dim=1), features[1]))
            features_1 = self._embedding((torch.cat([features[2], message_1], dim=1), features[3]))
            return self._merge(features[0], features_0), features[1], self._merge(features[2], features_1), features[3]

        message_0 = self._attn(features[0], features[1], features[1])
        message_1 = self._attn(features[1], features[0], features[0])

        features_0 = self._embedding(torch.cat([features[0], message_0], dim=1))
        features_1 = self._embedding(torch.cat([features[1], message_1], dim=1))
        return self._merge(features[0], features_0), self._merge(features[1], features_1)


########################################################################################################################
# GPT-2 Style Transformer Blocks
class GPTDualAttentionPropagation(GraphDualAttentionPropagation):
    """
    GPT(2)-style attentional propagation layer base implementation for dual graph self/cross attention

    """

    # GPT style MLP embedding configuration
    def _embedding_default(self, dim: int):
        return {
            'input_dimension': dim,
            'feature_dimension': dim,
            'channels': [dim, dim * 4, dim],
            'masked': self._config['masked'],
            'norm': None,
            'init': 'all',                      # initialize all bias weights to 0.0
        }

    def __init__(self, config: dict):
        super(GPTDualAttentionPropagation, self).__init__(config)

        if 'pre_norm_attention' in self._config and self._config['pre_norm_attention'] is not None:
            if isinstance(self._config['pre_norm_attention'], str):
                norm_config = {'feature_dimension': self._config['feature_dimension'], 'masked': self._config['masked']}
            elif isinstance(self._config['pre_norm_attention'], dict):
                norm_config = self._config['pre_norm_attention']
                norm_config['feature_dimension'] = self._config['feature_dimension']
                norm_config['masked'] = self._config['masked']
            else:
                raise ValueError(f'expected pre normalization configuration of attention module to be of type '
                                 f'[str, dict], but is {type(self._config["pre_norm_attention"])}')
            self._pre_norm_attention = AttentionBatchNorm(norm_config)
            self._config['pre_norm_attention'] = self._pre_norm_attention.config
        else:
            self._pre_norm_attention = lambda x: x

        if 'pre_norm_embedding' in self._config and self._config['pre_norm_embedding'] is not None:
            if isinstance(self._config['pre_norm_embedding'], str):
                norm_config = {'feature_dimension': self._config['feature_dimension'], 'masked': self._config['masked']}
            elif isinstance(self._config['pre_norm_embedding'], dict):
                norm_config = self._config['pre_norm_embedding']
                norm_config['feature_dimension'] = self._config['feature_dimension']
                norm_config['masked'] = self._config['masked']
            else:
                raise ValueError(f'expected pre normalization configuration of attention module to be of type '
                                 f'[str, dict], but is {type(self._config["pre_norm_embedding"])}')
            self._pre_norm_embedding = AttentionBatchNorm(norm_config)
            self._config['pre_norm_embedding'] = self._pre_norm_embedding.config
        else:
            self._pre_norm_embedding = lambda x: x

    @abstractmethod
    def forward(self, features: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        pass


class GPTSelfAttentionPropagation(GPTDualAttentionPropagation):
    def __init__(self, config: dict):
        """
        GPT(2)-style attentional propagation layer for dual single source/self attention

        """
        super(GPTSelfAttentionPropagation, self).__init__(config)
        self._config['type'] = 'gpt-self'

    def forward(self, features):
        encoding_0 = self._encoding(0) if self._encoding is not None else None
        encoding_1 = self._encoding(1) if self._encoding is not None else None
        if self._config['masked']:
            input_0, _, input_1, _ = self._pre_norm_attention(features)
            residual_0 = features[0] + self._attn(input_0, input_0, input_0, mask=features[1], encoding=encoding_0)
            features_0 = self._embedding(self._pre_norm_embedding((residual_0, features[1])))
            residual_1 = features[2] + self._attn(input_1, input_1, input_1, mask=features[3], encoding=encoding_1)
            features_1 = self._embedding(self._pre_norm_embedding((residual_1, features[3])))
            return self._merge(residual_0, features_0), features[1], self._merge(residual_1, features_1), features[3]

        input_0 = self._pre_norm_attention(features[0])
        input_1 = self._pre_norm_attention(features[1])
        residual_0 = features[0] + self._attn(input_0, input_0, input_0, encoding=encoding_0)
        features_0 = self._embedding(self._pre_norm_embedding(residual_0))
        residual_1 = features[1] + self._attn(input_1, input_1, input_1, encoding=encoding_1)
        features_1 = self._embedding(self._pre_norm_embedding(residual_1))
        return self._merge(residual_0, features_0), self._merge(residual_1, features_1)


class GPTCrossAttentionPropagation(GPTDualAttentionPropagation):
    def __init__(self, config: dict):
        """
        GPT(2)-style attentional propagation layer for cross attention

        """
        super(GPTCrossAttentionPropagation, self).__init__(config)
        self._config['type'] = 'gpt-cross'

    def forward(self, features):
        if self._config['masked']:
            input_0, _, input_1, _ = self._pre_norm_attention(features)
            residual_0 = features[0] + self._attn(input_0, input_1, input_1, mask=(features[1], features[3]))
            residual_1 = features[2] + self._attn(input_1, input_0, input_0, mask=(features[3], features[1]))

            features_0 = self._embedding(self._pre_norm_embedding((residual_0, features[1])))
            features_1 = self._embedding(self._pre_norm_embedding((residual_1, features[3])))
            return self._merge(residual_0, features_0), features[1], self._merge(residual_1, features_1), features[3]

        input_0 = self._pre_norm_attention(features[0])
        input_1 = self._pre_norm_attention(features[1])
        residual_0 = features[0] + self._attn(input_0, input_1, input_1)
        residual_1 = features[1] + self._attn(input_1, input_0, input_0)

        features_0 = self._embedding(self._pre_norm_embedding(residual_0))
        features_1 = self._embedding(self._pre_norm_embedding(residual_1))
        return self._merge(residual_0, features_0), self._merge(residual_1, features_1)


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
        'masked': False,                    # use masked attention
        # 'norm": {type: 'group'},          # optional default configuration for normalization layers
    }

    # default norm
    _default_norm = {
        'type': 'batch',        # batch, group, layer
        'dims': 1,              # 1D batch norm
        'eps': 1e-6,
        'masked': False,
    }

    def __init__(self, config: dict, encoding: Callable[[int], torch.Tensor] = None):
        super(GraphCrossAttentionNetwork, self).__init__()
        self._config = merge_dict(self._default, config)

        modules = nn.ModuleList()
        modules_config = []
        attention_propagation_default = {
            'feature_dimension': self._config['feature_dimension'],
            'heads': self._config['heads'],
            'merge': self._config['merge'],
            'masked': self._config['masked'],
        }
        gpt_attention_default = {
            'pre_norm_attention': 'norm',
            'pre_norm_embedding': 'norm',
        }
        self._default_norm['masked'] = self._config['masked']
        if 'norm' in self._config:
            self._config['norm']['masked'] = self._config['masked']
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
                    modules[-1].encoding(encoding)

                elif layer == 'gpt-cross':
                    # cross attention propagation layer
                    modules.append(GPTCrossAttentionPropagation(
                        {**attention_propagation_default, **gpt_attention_default}))

                elif layer == 'gpt-self':
                    # self attention propagation layer
                    modules.append(GPTSelfAttentionPropagation(
                        {**attention_propagation_default, **gpt_attention_default}))
                    modules[-1].encoding(encoding)

                else:
                    raise RuntimeError(f'unknown graph attention module: \"{layer}\"')

            elif isinstance(layer, dict) and 'type' in layer:
                # configuration dictionary for layer
                layer['feature_dimension'] = self._config['feature_dimension']
                layer['masked'] = self._config['masked']
                if layer['type'] == 'batch':
                    modules.append(AttentionBatchNorm(layer))

                elif layer['type'] == 'cross':
                    layer_config = merge_dict(attention_propagation_default, layer)
                    modules.append(GraphCrossAttentionPropagation(layer_config))

                elif layer['type'] == 'self':
                    layer_config = merge_dict(attention_propagation_default, layer)
                    modules.append(GraphSelfAttentionPropagation(layer_config))
                    modules[-1].encoding(encoding)

                elif layer['type'] == 'gpt-cross':
                    layer_config = merge_dict(attention_propagation_default, layer)
                    modules.append(GPTCrossAttentionPropagation(layer_config))

                elif layer['type'] == 'gpt-self':
                    layer_config = merge_dict(attention_propagation_default, layer)
                    modules.append(GPTSelfAttentionPropagation(layer_config))
                    modules[-1].encoding(encoding)

                else:
                    raise RuntimeError(f'unknown layer type {layer["type"]}')

            else:
                raise RuntimeError(f'unknown layer definition of type {type(layer)}')

            modules_config.append(modules[-1].config)

        self._config['layers'] = modules_config
        self._model = nn.Sequential(*modules)

    def forward(self, features: tuple[torch.Tensor, ...]):
        res = self._model(features)
        return (res[0], res[2]) if self._config['masked'] else res

    @property
    def config(self) -> dict:
        return self._config
