"""
 adaptions as feature head

Ludwig Mohr
ludwig.mohr@icg.tugraz.at

"""
import torch
from torch import nn
import torch.nn.functional
from typing import Union

from .neighbourhood import StaticNeighbourhood
from utils.misc import merge_dict_default, merge_dict
from .misc import get_layers_geometric


################################################################################
class FeatureDGCNN(nn.Module):
    """
    DGCNN adaption as point feature head

    """
    _default = {
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

        if neighbourhood:
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

    def _get_graph_feature(self, data: torch.Tensor, idx: int = None):
        batch_size, num_dims, num_points = data.shape
        # data = data.view(batch_size, -1, num_points)  # from original code. why is it here? what does it do?
        idx = self._neighbourhood(data, idx=idx)   # (batch_size, num_points, k)

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
            self._neighbourhood.recompute(x, idx=idx)
        elif idx:
            self._neighbourhood.idx = idx

        # encoding
        intermediate = []
        for layer in self._encoder:
            x = self._get_graph_feature(x, idx=idx)
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


################################################################################
class MLP(nn.Module):
    """
    Multi-layer perceptron, taken from SuperGlue code -> class version
    PointNet-like per point MLP, without the MaxPool/aggregation

    """
    _default = {
        'input_dimension': 3,
        'feature_dimension': 32,
        'channels': None,
        'layers': 4,
        'norm': 'batch',            # batch, layer, none
        'bias': True,
    }

    def __init__(self, config: dict):
        super(MLP, self).__init__()

        self._config = merge_dict(self._default, config)
        self._layers()

        num_layers = self._config['layers']
        channels = self._config['channels']
        layers = []
        norm = self._config['norm'] is not None and self._config['norm'] in ['batch', 'layer']
        for i in range(0, num_layers):
            layers.append(
                nn.Conv1d(channels[i], channels[i+1], kernel_size=1, bias=self._config['bias']))
            if i < (num_layers - 2):
                if norm and self._config['norm'] == 'batch':
                    layers.append(nn.BatchNorm1d(channels[i+1]))
                elif norm and self._config['norm'] == 'layer':
                    layers.append(nn.GroupNorm(1, channels[i+1]))
                layers.append(nn.ReLU())
        self._nn = nn.Sequential(*layers)

    def forward(self, data: Union[torch.Tensor, list[torch.Tensor], tuple[torch.Tensor]], idx: int = None):
        if isinstance(data, torch.Tensor):
            return self._nn(data)
        else:
            # assume iterable[torch.Tensor]
            return (self._nn(element) for element in data)

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
