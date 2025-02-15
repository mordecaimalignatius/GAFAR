"""
 adaptions as feature head

Ludwig Mohr
ludwig.mohr@icg.tugraz.at

"""
import torch
from torch import nn
import torch.nn.functional
from typing import Union
from abc import ABC

from .neighbourhood import StaticNeighbourhood
from utils.misc import merge_dict_default, merge_dict
from .misc import get_layers_geometric
from .norm import MaskedBatchNorm1d, MaskedBatchNorm2d


########################################################################################################################
class FeatureDGCNN(nn.Module):
    """
    DGCNN adaption as point feature head

    """
    _default = {
        'type': 'knn',
        'input_dimension': 3,           # dimension of input features
        'feature_dimension': 32,        # dimension of output features
        'relu_slope': 0.2,              # slope of leaky relu
        'embedding_dimension': 1024,    # maximum dimension for embedding
        'encoder': [64, 64, 128, 256],  # list of encoder layer dimensions.
                                        # if not given, will be calculated.
        'embedding': [512, 128],        # list of decoder layers (for feature reduction).
                                        # calculated if not given
        'embedding_bias': True,         # use bias vector in embedding layers
        'encoder_start_dimension': 64,  # output dimension of first encoder layer (layer dimensions not calculated
                                        # from input_dimension), N-1 layers. if None, N layers from input_dimension
        'encoder_layers': 5,            # number of layers in feature encoder
        'embedding_layers': 3,          # number of layers in feature embedding bottleneck
        'dropout': False,               # use dropout in embedding layers
        'dropout_rate': 0.5,            # probability of a neuron getting dropped
        'neighbourhood': {              # default point neighbourhood. may be config dict or neighbourhood instance
            'size': 20,
            'length': 1,
            'grad': False,
        },
        'norm': 'batch',                # batch, layer
    }

    def __init__(self, config: dict, neighbourhood: StaticNeighbourhood = None):
        super(FeatureDGCNN, self).__init__()
        self._config = merge_dict_default(self._default, config)

        if neighbourhood is not None:
            self._neighbourhood = neighbourhood
            self._manage_neighbourhood = False
        elif isinstance(self._config['neighbourhood'], StaticNeighbourhood):
            self._neighbourhood = neighbourhood
            self._manage_neighbourhood = False
        else:
            self._neighbourhood = StaticNeighbourhood(
                self._config['neighbourhood']['size'],
                length=self._config['neighbourhood']['length'],
                grad=self._config['neighbourhood']['grad'],
            )
            self._manage_neighbourhood = True
        self._k = self._neighbourhood.k

        # encoder/embedding/decoder layer dimensions
        self._layer_dimensions()

        encoder = self._config['encoder']
        norm = self._config['norm']
        self._encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2*encoder[idx-1], encoder[idx], kernel_size=1, bias=False),
                nn.BatchNorm2d(encoder[idx]) if norm == 'batch' else nn.GroupNorm(1, encoder[idx]),
                nn.LeakyReLU(negative_slope=self._config['relu_slope'])
            ) for idx in range(1, len(encoder)-1)]
        )
        self._encoder_last = nn.Sequential(
            nn.Conv1d(sum(encoder[1:-1]), encoder[-1], kernel_size=1, bias=False),
            nn.BatchNorm1d(encoder[-1]) if norm == 'batch' else nn.GroupNorm(1, encoder[-1]),
            nn.LeakyReLU(negative_slope=self._config['relu_slope'])
        )

        # bottleneck / feature encoding layers
        embedding = self._config['embedding']
        self._feature = nn.Sequential(*[
            (nn.Sequential(
                nn.Conv1d(embedding[idx-1], embedding[idx], kernel_size=1, bias=self._config['embedding_bias']),
                nn.BatchNorm1d(embedding[idx]) if norm == 'batch' else nn.GroupNorm(1, embedding[idx]),
                nn.LeakyReLU(negative_slope=self._config['relu_slope']),
                nn.Dropout(self._config['dropout_rate'])
            ) if self._config['dropout']
             else nn.Sequential(
                 nn.Conv1d(embedding[idx - 1], embedding[idx], kernel_size=1, bias=self._config['embedding_bias']),
                 nn.BatchNorm1d(embedding[idx]) if norm == 'batch' else nn.GroupNorm(1, embedding[idx]),
                 nn.LeakyReLU(negative_slope=self._config['relu_slope']))
             ) if idx < (len(embedding)-1)
            else nn.Conv1d(
                embedding[idx-1], embedding[idx], kernel_size=1, bias=self._config['embedding_bias'])
            for idx in range(1, len(embedding))])

    def _get_graph_feature(self, data: torch.Tensor, idx: int = None, ndim: int = None):
        batch_size, num_dims, num_points = data.shape
        # data = data.view(batch_size, -1, num_points)  # from original code. why is it here? what does it do?
        if ndim is None:
            idx = self._neighbourhood(data, idx=idx)   # (batch_size, num_points, k)
        else:
            idx = self._neighbourhood(data[:, :ndim, :], idx=idx)

        idx_base = torch.arange(0, batch_size, device=data.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)

        # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
        # batch_size * num_points * k + range(0, batch_size*num_points)
        data = data.transpose(2, 1).contiguous()
        feature = data.view(batch_size*num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, self._k, num_dims)
        data = data.view(batch_size, num_points, 1, num_dims).repeat(1, 1, self._k, 1)

        feature = torch.cat((feature-data, data), dim=3).permute(0, 3, 1, 2).contiguous()

        return feature

    def forward(self, x: torch.Tensor, idx: int = None):
        # x: data torch.Tensor (Batch x Dimension (3, 6 w/ normals) x Points
        # normalize points
        # x = x.transpose(2, 1)
        if self._manage_neighbourhood:
            self._neighbourhood.recompute(x[:, :3, :].contiguous(), idx=idx)

        # encoding
        intermediate = []
        ndim = 3
        for layer in self._encoder:
            x = self._get_graph_feature(x, idx=idx, ndim=ndim)
            ndim = None     # in first layer ignore normals if present
            x = layer(x).max(dim=-1, keepdim=False)[0]
            intermediate.append(x)

        x = torch.cat(intermediate, dim=1)
        x = self._encoder_last(x)

        # bottleneck
        x = self._feature(x)

        return x

    def _layer_dimensions(self):
        """ make sure layer dimensions are set and consistent """
        # get encoder layer dimensions
        if not self._config['encoder']:
            # calculate
            layers = get_layers_geometric(
                self._config['encoder_start_dimension'] if self._config['encoder_start_dimension']
                else self._config['input_dimension'],
                self._config['embedding_dimension'],
                self._config['encoder_layers'] - 1 if self._config['encoder_start_dimension']
                else self._config['encoder_layers'],
            )
            self._config['encoder'] = \
                [self._config['input_dimension'], *layers] if self._config['encoder_start_dimension'] else layers

        else:
            # make sure info is correct
            # update length of encoder layers
            if self._config['encoder'][0] != self._config['input_dimension']:
                self._config['encoder'].insert(0, self._config['input_dimension'])
            if self._config['encoder'][-1] != self._config['embedding_dimension']:
                self._config['encoder'].append(self._config['embedding_dimension'])

            self._config['encoder_layers'] = len(self._config['encoder']) - 1

        # get feature embedding dimensions
        if not self._config['embedding']:
            # calculate
            self._config['embedding'] = get_layers_geometric(
                self._config['embedding_dimension'],
                self._config['feature_dimension'],
                self._config['embedding_layers'],
            )

        else:
            if self._config['embedding'][0] != self._config['embedding_dimension']:
                self._config['embedding'].insert(0, self._config['embedding_dimension'])
            if self._config['embedding'][-1] != self._config['feature_dimension']:
                self._config['embedding'].append(self._config['feature_dimension'])
            self._config['embedding_layers'] = len(self._config['embedding']) - 1

    @property
    def config(self) -> dict:
        return self._config

    def update_radius(self, r: float):
        # compatibility
        pass


########################################################################################################################
class FeatureRadiusDGCNN(nn.Module):
    _default = {
        'type': 'radius',
        'input_dimension': 3,  # dimension of input features
        'feature_dimension': 32,  # dimension of output features
        'relu_slope': 0.2,  # slope of leaky relu
        'embedding_dimension': 1024,  # maximum dimension for embedding
        'encoder': [64, 64, 128, 256],  # list of encoder layer dimensions.
        # if not given, will be calculated.
        'embedding': [512, 128],  # list of decoder layers (for feature reduction).
        # calculated if not given
        'embedding_bias': True,  # use bias vector in embedding layers
        'encoder_start_dimension': 64,  # output dimension of first encoder layer (layer dimensions not calculated
        # from input_dimension), N-1 layers. if None, N layers from input_dimension
        'encoder_layers': 5,  # number of layers in feature encoder
        'embedding_layers': 3,  # number of layers in feature embedding bottleneck
        'dropout': False,  # use dropout in embedding layers
        'dropout_rate': 0.5,  # probability of a neuron getting dropped
        'neighbourhood': {  # default point neighbourhood. may be config dict or neighbourhood instance
            'method': 'radius',
            'size': 20,  # maximum number of neighbours to consider
            'max': 32,  # maximum number of neighbours to query in radius search (if applicable)
            'radius': 0.1,
            'min': 1,  # minimum number of neighbours to consider feature calculation valid
            'length': 1,
            'grad': False,
        },
        'norm': 'batch',  # batch, layer
    }

    def __init__(self, config: dict, neighbourhood: StaticNeighbourhood = None):
        super(FeatureRadiusDGCNN, self).__init__()
        self._config = merge_dict_default(self._default, config)

        if neighbourhood is not None:
            self._neighbourhood = neighbourhood
            self._manage_neighbourhood = False
        elif isinstance(self._config['neighbourhood'], StaticNeighbourhood):
            self._neighbourhood = neighbourhood
            self._manage_neighbourhood = False
        else:
            n_conf = self._config['neighbourhood']
            self._neighbourhood = StaticNeighbourhood(
                n_conf['size'],
                method=n_conf['method'],
                m=n_conf['max'],
                r=n_conf['radius'],
                length=n_conf['length'],
                grad=n_conf['grad'],
                order='bdn',
            )
            self._manage_neighbourhood = True
        self._k = self._neighbourhood.k

        # encoder/embedding/decoder layer dimensions
        self._layer_dimensions()

        encoder = self._config['encoder']
        self._encoder = nn.ModuleList([
            MaskedDGCNNLayer2D(2 * encoder[idx - 1], encoder[idx], self._config['norm'], self._config['relu_slope'])
            for idx in range(1, len(encoder) - 1)])
        self._encoder_last = MaskedDGCNNLayer1D(
            sum(encoder[1:-1]), encoder[-1], self._config['norm'], self._config['relu_slope'])

        # bottleneck / feature encoding layers
        embedding = self._config['embedding']
        self._feature = nn.ModuleList([
            MaskedDGCNNLayer1D(embedding[idx - 1], embedding[idx], self._config['norm'], self._config['relu_slope'],
                               self._config['dropout_rate'] if self._config['dropout'] else None,
                               self._config['embedding_bias']) if idx < (len(embedding) - 1)
            else MaskedConv1D(
                embedding[idx - 1], embedding[idx], bias=self._config['embedding_bias'])
            for idx in range(1, len(embedding))])

        self._timer = None
        self._valid = [None] * self._config['neighbourhood']['length']

    def _get_graph_feature(self, data: torch.Tensor, idx: int = None, ndim: int = None):
        batch_size, num_dims, num_points = data.shape
        if ndim is None:
            idx, valid = self._neighbourhood(data, idx=idx)  # (batch_size, num_points, k)
        else:
            idx, valid = self._neighbourhood(data[:, :ndim, :], idx=idx)

        idx_base = torch.arange(0, batch_size, device=data.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)

        # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
        # batch_size * num_points * k + range(0, batch_size*num_points)
        data = data.transpose(2, 1).contiguous()
        feature = data.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, self._k, num_dims)
        data = data.view(batch_size, num_points, 1, num_dims).repeat(1, 1, self._k, 1)

        feature = torch.cat((feature - data, data), dim=3)  # batch_size x num_points x K x 2x feature_dim
        feature = feature * valid.unsqueeze(-1)
        # feature = feature.where(
        #    valid.unsqueeze(-1).expand(-1, -1, -1, feature.shape[3]),
        #    torch.tensor(0.0, device=data.device))
        feature = feature.permute(0, 3, 1, 2).contiguous()  # batch_size x 2*feature_dim x num_points x K

        return feature, valid.unsqueeze(1)  # for MaskedBatchNorm2d batch_size x 1 x num_points x K

    def forward(self, x: torch.Tensor, idx: int = None):
        # x: data torch.Tensor (Batch x Dimension (3, 6 w/ normals) x Points
        # normalize points
        # x = x.transpose(2, 1)
        if self._manage_neighbourhood:
            self._neighbourhood.recompute(x[:, :3, :].contiguous(), idx=idx)

        # encoding
        intermediate = []
        all_valid = torch.ones((x.shape[0], 1, x.shape[2]), dtype=torch.bool, device=x.device)
        ndim = 3
        for layer in self._encoder:
            x, valid = self._get_graph_feature(x, idx=idx, ndim=ndim)
            ndim = None  # in first layer ignore normals if present
            x, _ = layer(x, valid)
            # set invalid values to -inf for torch.max(dim=-1) max-pooling
            x[torch.logical_not(valid).expand(-1, x.shape[1], -1, -1)] = -torch.inf
            x = x.max(dim=-1, keepdim=False)[0]
            x[torch.lt(valid.sum(-1), self._config['neighbourhood']['min']).expand(-1, x.shape[1], -1)] = 0.0
            intermediate.append(x)
            all_valid = torch.logical_and(all_valid, torch.ge(valid.sum(-1), self._config['neighbourhood']['min']))

        self._valid[idx] = all_valid
        x = torch.cat(intermediate, dim=1)
        x, _ = self._encoder_last(x, all_valid)

        # bottleneck
        for layer in self._feature:
            x, _ = layer(x, all_valid)

        return x

    def _layer_dimensions(self):
        """ make sure layer dimensions are set and consistent """
        # get encoder layer dimensions
        if not self._config['encoder']:
            # calculate
            layers = get_layers_geometric(
                self._config['encoder_start_dimension'] if self._config['encoder_start_dimension']
                else self._config['input_dimension'],
                self._config['embedding_dimension'],
                self._config['encoder_layers'] - 1 if self._config['encoder_start_dimension']
                else self._config['encoder_layers'],
            )
            self._config['encoder'] = \
                [self._config['input_dimension'], *layers] if self._config['encoder_start_dimension'] else layers

        else:
            # make sure info is correct
            # update length of encoder layers
            if self._config['encoder'][0] != self._config['input_dimension']:
                self._config['encoder'].insert(0, self._config['input_dimension'])
            if self._config['encoder'][-1] != self._config['embedding_dimension']:
                self._config['encoder'].append(self._config['embedding_dimension'])

            self._config['encoder_layers'] = len(self._config['encoder']) - 1

        # get feature embedding dimensions
        if not self._config['embedding']:
            # calculate
            self._config['embedding'] = get_layers_geometric(
                self._config['embedding_dimension'],
                self._config['feature_dimension'],
                self._config['embedding_layers'],
            )

        else:
            if self._config['embedding'][0] != self._config['embedding_dimension']:
                self._config['embedding'].insert(0, self._config['embedding_dimension'])
            if self._config['embedding'][-1] != self._config['feature_dimension']:
                self._config['embedding'].append(self._config['feature_dimension'])
            self._config['embedding_layers'] = len(self._config['embedding']) - 1

    @property
    def config(self) -> dict:
        return self._config

    def valid(self, idx: int = None):
        # return points with valid neighbourhood
        return self._valid[idx if idx is not None else self._neighbourhood.idx]

    def update_radius(self, r: float):
        # update search radius of neighbourhood
        self._neighbourhood.radius = r


class _MaskedDGCNNLayer(nn.Module, ABC):
    """
    masked single DGCNN layer

    """

    def __init__(self):
        super(_MaskedDGCNNLayer, self).__init__()

        self._conv = None
        self._norm = None
        self._non_lin = None
        self._do = None

    def forward(self, batch: torch.Tensor, mask: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        """
        batch: N x C x M (x K) data tensor
        mask: N x 1 x M (x K) mask tensor
        """

        x = self._conv(batch)
        x = self._non_lin(self._norm(x, mask))
        if self._do is not None:
            x = self._do(x)
        return x, mask


class MaskedDGCNNLayer1D(_MaskedDGCNNLayer):
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'batch', relu_slope: float = 0.2,
                 drop_out: float = None, bias: bool = False):
        super(MaskedDGCNNLayer1D, self).__init__()

        if norm != 'batch':
            raise RuntimeError('masked group norm not implemented')

        self._conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
        self._norm = MaskedBatchNorm1d(out_channels)  # if norm == 'batch' else nn.GroupNorm(1, out_channels)
        self._non_lin = nn.LeakyReLU(negative_slope=relu_slope)
        self._do = None
        if drop_out is not None:
            self._do = nn.Dropout(drop_out)


class MaskedDGCNNLayer2D(_MaskedDGCNNLayer):
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'batch', relu_slope: float = 0.2,
                 drop_out: float = None, bias: bool = False):
        super(MaskedDGCNNLayer2D, self).__init__()

        if norm != 'batch':
            raise RuntimeError('masked group norm not implemented')

        self._conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self._norm = MaskedBatchNorm2d(out_channels)  # if norm == 'batch' else nn.GroupNorm(1, out_channels)
        self._non_lin = nn.LeakyReLU(negative_slope=relu_slope)
        self._do = None
        if drop_out is not None:
            self._do = nn.Dropout(drop_out)


class MaskedConv1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(MaskedConv1D, self).__init__()
        self._conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        return self._conv(x), mask


########################################################################################################################
class MLP(nn.Module):
    """
    Multi-layer perceptron, taken from SuperGlue code -> class version
    PointNet-like per point MLP, without the MaxPool/aggregation

    """
    _default = {
        'type': 'mlp',
        'input_dimension': 3,
        'feature_dimension': 32,
        'channels': None,
        'layers': 4,
        'norm': 'batch',            # batch, layer, none
        'bias': True,
        'masked': False,
    }

    def __init__(self, config: dict):
        super(MLP, self).__init__()

        self._config = merge_dict(self._default, config)
        self._layers()
        
        masked = self._config['masked']
        num_layers = self._config['layers']
        channels = self._config['channels']
        layers = []
        norm = self._config['norm'] is not None and self._config['norm'] in ['batch', 'layer']
        for i in range(0, num_layers):
            layers.append([f'conv_{i}',
                           nn.Conv1d(channels[i], channels[i+1], kernel_size=1, bias=self._config['bias'])])
            if i < (num_layers - 1):
                if norm and self._config['norm'] == 'batch':
                    layers.append([f'norm_{i}',
                                   MaskedBatchNorm1d(channels[i+1]) if masked else nn.BatchNorm1d(channels[i+1])])
                elif norm and self._config['norm'] == 'layer':
                    layers.append([f'norm_{i}', nn.GroupNorm(1, channels[i+1])])
                layers.append([f'relu_{i}', nn.ReLU()])
        self._nn = nn.ModuleDict(layers)

        if 'init' in self._config:
            if self._config['init'] == 'last':
                self.init_constant(0.0)
            elif self._config['init'] == 'all':
                self.init_constant_all(0.0)
            elif self._config['init'] != 'none':
                raise ValueError(f'MLP bias initialization: unknown value \"{self._config["init"]}\"')

    def forward(self, data: Union[torch.Tensor, list[torch.Tensor], tuple[torch.Tensor]], idx: int = None):
        if isinstance(data, torch.Tensor):
            return self._forward(data)
        elif self._config['masked']:
            # interleaved tensor_0, mask_0, ... tensor_n, mask_n
            if len(data) > 2:
                raise RuntimeError(f'masked MLP for more than 1 input not implemented')
            return self._forward(data[0], mask=data[1])
        else:
            # assume iterable[torch.Tensor]
            return tuple([self._forward(element) for element in data])

    def _forward(self, data: torch.Tensor, mask: torch.Tensor = None):
        for key in self._nn.keys():
            masked = key.startswith('norm') and self._config['masked']
            data = self._nn[key](data, mask=mask) if masked else self._nn[key](data)

        return data

    def _layers(self):
        """ get/correct number of layers, correct metadata """
        if self._config['channels']:
            if self._config['channels'][0] != self._config['input_dimension']:
                self._config['channels'].insert(0, self._config['input_dimension'])
            if self._config['channels'][-1] != self._config['feature_dimension']:
                self._config['channels'].append(self._config['feature_dimension'])
            self._config['layers'] = len(self._config['channels']) - 1

        else:
            self._config['channels'] = get_layers_geometric(
                self._config['input_dimension'],
                self._config['feature_dimension'],
                self._config['layers'],
            )

    @property
    def config(self) -> dict:
        return self._config

    def init_constant(self, value: float):
        """ constant initialization of bias of last layer """
        torch.nn.init.constant_(self._nn[list(self._nn.keys())[-1]].bias, value)

    def init_constant_all(self, value: float):
        """ constant initialization of bias vectors of all Conv1d layers """
        for key in self._nn.keys():
            if not isinstance(self._nn[key], nn.Conv1d):
                continue
            torch.nn.init.constant_(self._nn[key].bias, value)
