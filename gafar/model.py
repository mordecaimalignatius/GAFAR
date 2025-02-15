"""
point matching network based on fully connected graph attention layers.
all input points in point set A will be matched with those of point set B.


Ludwig Mohr
ludwig.mohr@icg.tugraz.at

"""
import torch

from .attention import GraphCrossAttentionNetwork
from .misc import normalize_3d_points_sphere, log_optimal_transport
from .feature import FeatureDGCNN, FeatureRadiusDGCNN, MLP
from .encoding import LearnableFourierPositionalEncoding
from .neighbourhood import StaticNeighbourhood
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
        'type': 'GAFAR',
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


########################################################################################################################
class GAFARv2(torch.nn.Module):
    """
    Graph-Attention Feature Augmentation for Registration - improved
    Small Point Cloud Matching Deep Neural Network

    Given two sets of points (source point set M, BxMx3/6, reference point set BxNx3/6)
    correspondences are determined by:
      1. Point Encoding (normalization + feature encoding, rotational embedding)
      2. Fully connected Graph Neural Network for feature augmentation
         consisting of multiple self and cross-attention layers
      3. Final projection
      4. Matching score via feature similarity and matchability or Optimal Transport (Sinkhorn Algorithm)
      5. Calculation of point matching matrix, Thresholding matrix based on mutual exclusivity and a score threshold

    Points without (found) correspondence are indicated by -1 in the correspondence ids.

    """
    _default_config = {
        'type': 'GAFARv2',
        'feature_dimension': 64,
        'head': {
            'type': 'matchable',
        },
        # matching cross attention network
        'matcher': {
            'layers': ['self', 'cross'] * 9,
            'heads': 2,
            'merge': 'add',
            'masked': False,
        },
        # feature encoding
        'feature': {
            'type': 'radius',
            'encoder': {
                'channels': [8, 16, 32],
            },
            'attention': {
                'layers': 4,                # number of attention layers. formerly num_layers
                'heads': 2,                 # number of attention heads to use
            },
            'neighbourhood': {
                'size': 20,                 # neighbourhood size
                'length': 2,                # number of neighbourhoods (i.e. two for matching of two point sets)
                'grad': False,              # pass gradients through neighbourhood calculation
            },
        },
        'normals': False,
    }

    _default_optimal = {
        'type': 'optimal',
        'iterations': 10,
    }

    _default_radius_neighbourhood = {
        'type': 'static',
        'method': 'radius',
        'size': 20,
        'radius': 0.1,
        'length': 2,
        'min': 1,
        'grad': False,
    }

    def __init__(self, config: dict):
        super(GAFARv2, self).__init__()
        self._config = merge_dict(self._default_config, config)
        config = merge_dict_default(self._default_config, config)

        # neighbourhood calculation moved to matcher (call in forward)
        self._masked = False
        neighbour_conf = self._config['feature']['neighbourhood']
        if self._config['feature']['type'].endswith('radius'):
            neighbour_conf = {**self._default_radius_neighbourhood, **neighbour_conf}
            if not neighbour_conf['method'].startswith('radius'):
                neighbour_conf['method'] = 'radius'
            # use masked attention for radius neighbourhood
            self._config['matcher']['masked'] = True
            self._masked = True

            config['feature']['neighbourhood'] = neighbour_conf

        self._neighbourhood = StaticNeighbourhood(
            neighbour_conf['size'],
            r=neighbour_conf['radius'] if 'radius' in neighbour_conf else None,
            m=neighbour_conf['max'] if 'max' in neighbour_conf else None,
            method=neighbour_conf['method'], length=neighbour_conf['length'], grad=neighbour_conf['grad'],
            order='bnd')

        # input dimension
        input_dimension = 6 if self._config['normals'] else 3

        # point feature network
        self._config['feature']['feature_dimension'] = self._config['feature_dimension']
        self._config['feature']['input_dimension'] = input_dimension
        feature = FeatureRadiusDGCNN if self._config['feature']['type'].endswith('radius') else FeatureDGCNN
        self._feature_network = feature(self._config['feature'], neighbourhood=self._neighbourhood)
        config['feature'] = self._feature_network.config
        config['feature']['neighbourhood'] = neighbour_conf

        self._config['matcher']['feature_dimension'] = self._config['feature_dimension']

        self._norm = 1. / self._config['matcher']['feature_dimension'] ** .5

        self._position = LearnableFourierPositionalEncoding(
            input_dimension,
            self._config['feature_dimension'] // self._config['matcher']['heads'],
            self._config['feature_dimension'] // self._config['matcher']['heads'],
            )
        self._encoding = (None, None)

        self._gnn = GraphCrossAttentionNetwork(self._config['matcher'], encoding=self.encoding)
        config['matcher'] = self._gnn.config

        self._final_projection = torch.nn.Conv1d(
            self._config['matcher']['feature_dimension'], self._config['matcher']['feature_dimension'],
            kernel_size=1, bias=True)
        self._matchability = None

        # set prediction head
        if self._config['head']['type'] == 'optimal':
            bin_score = torch.nn.Parameter(torch.tensor(1.))
            self.register_parameter('bin_score', bin_score)
            self.head = self.optimal_transport_head
            config['head'] = merge_dict(self._default_optimal, config["head"])

        elif self._config['head']['type'] == 'matchable':
            self._matchability = torch.nn.Conv1d(
                self._config['matcher']['feature_dimension'], 1, kernel_size=1, bias=True)
            self.head = self.matchability_head

        else:
            raise ValueError(f'unknown prediction head type \"{self._config["head"]["type"]}\"')

        # store cleaned up and full config (i.e. unknown parameters are removed)
        self._config = config
        self._valid = (None, None)

    def forward(self, points_0: torch.Tensor, points_1: torch.Tensor) -> dict:
        """ run feature extraction, point position encoding and matching on a pair of point sets """
        # update neighbourhood
        self._neighbourhood.recompute(points_0[..., :3], idx=0)
        self._neighbourhood.recompute(points_1[..., :3], idx=1)

        # point normalization.
        points_0, points_1 = points_0.clone(), points_1.clone()
        points_0[..., :3], points_1[..., :3], _ = normalize_3d_points_sphere(points_0[..., :3], points_1[..., :3])

        # positional encoding
        self._encoding = [self._position(points_0), self._position(points_1)]

        points_0 = points_0.transpose(1, 2).contiguous()
        points_1 = points_1.transpose(1, 2).contiguous()

        # feature network
        points_0_embedding = self._feature_network(points_0, idx=0)
        points_1_embedding = self._feature_network(points_1, idx=1)
        if getattr(self, '_masked', False):
            self._valid = (self._feature_network.valid(0), self._feature_network.valid(1))

        # Multi-layer Transformer network.
        if self._masked:
            features = (points_0_embedding, self._valid[0], points_1_embedding, self._valid[1])
        else:
            features = (points_0_embedding, points_1_embedding)
        points_0_embedding, points_1_embedding = self._gnn(features)

        # Final MLP projection.
        desc0, desc1 = self._final_projection(points_0_embedding), self._final_projection(points_1_embedding)

        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', desc0, desc1)
        scores = scores * self._norm

        # matcher head
        scores = self.head(scores, points_0_embedding, points_1_embedding)

        ret = {'score': scores}
        if self._masked:
            ret['valid0'] = self._valid[0].squeeze(1)
            ret['valid1'] = self._valid[1].squeeze(1)
        return ret

    def optimal_transport_head(self, scores: torch.Tensor, *args) -> torch.Tensor:
        # Run the optimal transport.
        if self._masked:
            mask = torch.logical_and(
                torch.logical_not(self._valid[0]).unsqueeze(2),
                torch.logical_not(self._valid[1]).unsqueeze(1),
            )
            scores[mask] = -100

        scores = log_optimal_transport(
            scores, self.bin_score,
            iters=self._config['head']['iterations'])

        return scores

    def matchability_head(self, score: torch.Tensor, desc0: torch.Tensor, desc1: torch.Tensor):
        """ matchability prediction head, adapted from LightGlue """
        b, m, n = score.shape
        match0 = self._matchability(desc0)
        match1 = self._matchability(desc1)

        # not_valid0 = not_valid1 = None
        if self._masked:
            # mask invalid values
            not_valid0 = torch.logical_not(self._valid[0])
            not_valid1 = torch.logical_not(self._valid[1])

            match0[not_valid0] = -100.0
            match1[not_valid1] = -100.0

            score0 = score.clone()
            score0[not_valid1.expand(-1, m, -1)] = -torch.inf
            score1 = score.clone()
            score1[not_valid0.view(b, n, 1).expand(-1, -1, n)] = -torch.inf
        else:
            score0 = score1 = score

        score0 = torch.nn.functional.log_softmax(score0, 2)
        # transpose -> log_softmax -> transpose for speed
        score1 = torch.nn.functional.log_softmax(score1.transpose(2, 1).contiguous(), 2).transpose(2, 1)

        scores = score.new_full((b, m+1, n+1), 0)
        scores[:, :-1, :-1] = score0 + score1 + \
            torch.nn.functional.logsigmoid(match0).view(b, m, 1) + \
            torch.nn.functional.logsigmoid(match1).view(b, 1, n)
        # s(-x) = 1 - s(x)
        scores[:, :-1, -1] = torch.nn.functional.logsigmoid(-match0).view(b, m)
        scores[:, -1, :-1] = torch.nn.functional.logsigmoid(-match1).view(b, n)

        return scores

    @property
    def config(self) -> dict:
        return self._config

    @property
    def last_layer(self) -> torch.nn.Module:
        return self._final_projection

    @property
    def last_layer_grad(self) -> torch.Tensor:
        return self._final_projection.weight.grad

    @property
    def valid(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._valid

    def update_radius(self, r: float = None) -> float:
        if r is not None:
            # update search radius of neighbourhood
            self._config['feature']['neighbourhood']['radius'] = r
            self._neighbourhood.radius = r

        return self._neighbourhood.radius

    def encoding(self, idx: int) -> torch.Tensor:
        # return encoding
        return self._encoding[idx]
