"""
point matching network based on fully connected graph attention layers.
all input points in point set A will be matched with those of point set B.


Ludwig Mohr
ludwig.mohr@icg.tugraz.at

"""
import torch

from .attention import GraphCrossAttentionNetwork
from .misc import normalize_3d_points_sphere, log_optimal_transport
from .feature import FeatureDGCNN, MLP
from utils.misc import merge_dict_default, merge_dict


################################################################################
class GAFAR(torch.nn.Module):
    """
    Graph-Attention Feature Augmentation for Registration
    Small Point Cloud Matching Deep Neural Network

    Given two sets of points (source point set M, BxMx3/6, reference point set BxNx3/6)
    correspondences are determined by:
      1. Point Encoding (normalization + position encoding and feature encoding, feature fusion)
      2. Fully connected Graph Neural Network for feature augmentation
         consisting of multiple self and cross-attention layers
      3. Final projection
      4. Optimal Transport (Sinkhorn Algorithm)
      5. Calculation of point matching matrix, Thresholding matrix based on mutual exclusivity and a score threshold

    Points without (found) correspondence are indicated by -1 in the correspondence ids.

    """
    _default_config = {
        'feature_dimension': 64,
        'early_fusion': 'mlp',      # [cat, add, mlp]
        'sinkhorn_iterations': 10,
        # matching cross attention network
        'matcher': {
            'layers': ['self', 'cross'] * 9,
            'heads': 2,
            'merge': 'add',
        },
        # position encoding
        'encoder': {
            'type': 'mlp',
            'channels': [8, 16, 32],
        },
        'feature': {
            'encoder': {
                'channels': [8, 16, 32],
            },
            'attention': {
                'layers': 4,                # number of attention layers. formerly num_layers
                'heads': 2,                 # number of attention heads to use
            },
            'neighbourhood': {
                'size': 20,             # neighbourhood size
                'length': 2,            # number of neighbourhoods (i.e. two for matching of two point sets)
                'grad': False,          # pass gradients through neighbourhood calculation
            },
        },
        'fusion': None,
        'normals': False,
    }

    def __init__(self, config: dict):
        super(GAFAR, self).__init__()
        self._config = merge_dict(self._default_config, config)
        config = merge_dict_default(self._default_config, config)

        # point/location encoder
        self._config['encoder']['feature_dimension'] = self._config['feature_dimension']
        self._config['encoder']['input_dimension'] = 6 if self._config['normals'] else 3
        self._point_encoder = MLP(self._config['encoder'])
        config['encoder'] = self._point_encoder.config

        # point feature network
        self._config['feature']['feature_dimension'] = self._config['feature_dimension']
        self._config['feature']['input_dimension'] = 6 if self._config['normals'] else 3
        self._feature_network = FeatureDGCNN(self._config['feature'])
        config['feature'] = self._feature_network.config

        # early fusion for point encoder and feature network output
        self._early_fusion_net = None
        if self._config['early_fusion'] == 'cat':
            self._early_fusion = self._early_fusion_cat
            self._config['matcher']['feature_dimension'] = 2 * self._config['feature_dimension']

        elif self._config['early_fusion'] == 'mlp':
            self._early_fusion = self._early_fusion_mlp
            self._config['matcher']['feature_dimension'] = self._config['feature_dimension']
            dimension = self._config['feature_dimension']
            fusion_default = {
                'channels': [2 * dimension, 2 * dimension, dimension],
                'norm': 'batch',
            }
            if self._config['fusion'] is None:
                self._config['fusion'] = fusion_default
            else:
                self._config['fusion'] = merge_dict(fusion_default, self._config['fusion'])
            self._config['fusion']['input_dimension'] = 2 * dimension
            self._config['fusion']['feature_dimension'] = dimension
            self._early_fusion_net = MLP(self._config['fusion'])
            config['fusion'] = self._early_fusion_net.config
        else:
            self._early_fusion = self._early_fusion_add
            self._config['matcher']['feature_dimension'] = self._config['feature_dimension']

        self._norm = 1. / self._config['matcher']['feature_dimension'] ** .5

        self._gnn = GraphCrossAttentionNetwork(self._config['matcher'])
        config['matcher'] = self._gnn.config

        self._final_projection = torch.nn.Conv1d(
            self._config['matcher']['feature_dimension'], self._config['matcher']['feature_dimension'],
            kernel_size=1, bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

        # store cleaned up and full config (i.e. unknown parameters are removed)
        self._config = config

    @staticmethod
    def _early_fusion_cat(data_0: torch.Tensor, data_1: torch.Tensor) -> torch.Tensor:
        return torch.cat((data_0, data_1), dim=1)

    @staticmethod
    def _early_fusion_add(data_0: torch.Tensor, data_1: torch.Tensor) -> torch.Tensor:
        return data_0 + data_1

    def _early_fusion_mlp(self, data_0: torch.Tensor, data_1: torch.Tensor) -> torch.Tensor:
        return self._early_fusion_net(torch.cat((data_0, data_1), dim=1))

    def forward(self, points_0: torch.Tensor, points_1: torch.Tensor) -> dict:
        """ run feature extraction, point position encoding and matching on a pair of point sets """

        # point normalization.
        points_0, points_1 = points_0.clone(), points_1.clone()
        points_0[..., :3], points_1[..., :3], _ = normalize_3d_points_sphere(points_0[..., :3], points_1[..., :3])
        points_0 = points_0.transpose(1, 2).contiguous()
        points_1 = points_1.transpose(1, 2).contiguous()

        # Keypoint MLP encoder.
        points_0_embedding = self._point_encoder(points_0)
        points_1_embedding = self._point_encoder(points_1)

        # feature network
        # the feature network is expected to manage its neighbourhood itself
        # self._feature_network.recompute_neighbourhood((points_0, points_1))
        points_0_embedding = self._early_fusion(points_0_embedding, self._feature_network(points_0, idx=0))
        points_1_embedding = self._early_fusion(points_1_embedding, self._feature_network(points_1, idx=1))

        # Multi-layer Transformer network.
        points_0_embedding, points_1_embedding = self._gnn(points_0_embedding, points_1_embedding)

        # Final MLP projection.
        desc0, desc1 = self._final_projection(points_0_embedding), self._final_projection(points_1_embedding)

        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', desc0, desc1)
        scores = scores * self._norm

        # Run the optimal transport.
        scores = log_optimal_transport(
            scores, self.bin_score,
            iters=self._config['sinkhorn_iterations'])

        return {'scores': scores}

    @property
    def config(self) -> dict:
        return self._config

    @property
    def last_layer(self) -> torch.nn.Module:
        return self._final_projection

    @property
    def last_layer_grad(self) -> torch.Tensor:
        return self._final_projection.weight.grad
