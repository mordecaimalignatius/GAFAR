"""
neighbourhood functions for graph networks

Ludwig Mohr
ludwig.mohr@icg.tugraz.at

"""
import torch
from torch_cluster import knn
from abc import ABC, abstractmethod
from contextlib import nullcontext

from typing import Optional, Union, List


################################################################################
class GraphNeighbourhood(ABC):
    def __init__(self, k, method: str = 'direct', grad: bool = False, **kwargs):
        """
        return the indices of k nearest neighbours

        :param method: cluster: use torch cluster, nn: use direct implementation ['nn', 'cluster']
        :param k:
        :param grad: pass gradients through KNN calculations
        :param kwargs:
        """
        self._k = k
        self._context = nullcontext() if grad else torch.no_grad()

        if method == 'cluster':
            self.knn = self._knn_cluster

        elif method == 'direct':
            self.knn = self._knn_direct

        else:
            raise RuntimeError(f'unknown knn algorithm: {method}')

        self._idx = 0
        self._len = 1

    def _knn_cluster(self, features: torch.Tensor):
        """
        calculate the k nearest neighbours for each feature

        :param features: BxDxN features
        :return:
        """
        with self._context:
            features = features.transpose(1, 2).contiguous()
            batch_size, num_points, feature_dim = features.shape
            device = features.device
            batch = torch.mul(
                torch.ones(features.shape[:2], dtype=torch.long, device=device),
                torch.arange(batch_size, dtype=torch.long, device=device).unsqueeze(1).expand(-1, num_points),
            )

            # 'for each point in y the k nearest points in x' knn(x, y, b_x, b_y, k)
            index = knn(features.view(-1, feature_dim), features.view(-1, feature_dim),
                        batch_x=batch.flatten(), batch_y=batch.flatten(),
                        k=self._k + 1)
            index = index[1, :].view(batch_size, num_points, self._k + 1)[:, :, 1:].contiguous()
            index -= torch.arange(0, batch_size * num_points, num_points,
                                  device=device).reshape(-1, 1, 1).expand(-1, num_points, self._k)

        return index.contiguous()

    def _knn_direct(self, features: torch.Tensor):
        """
        direct/greedy implementation of knn search. for small number of points/feature dimensions usually faster
        than torch_cluster.knn on gpu

        :param features: BxDxN feature vector
        :return:
        """
        with self._context:
            pairwise_inner = -2 * torch.matmul(features.transpose(1, 2), features)
            features_2 = torch.einsum('bdn,bdn->bn', features, features)

            distance = features_2.unsqueeze(1) + pairwise_inner + features_2.unsqueeze(2)

            _, index = distance.topk(k=self._k + 1, largest=False, dim=2)
            index = index[..., 1:].contiguous()

        return index

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        return the current nearest neighbour indices

        :return: BxNxK torch.long tensor
        """
        pass

    @abstractmethod
    def recompute(self, data: Union[torch.Tensor, List[torch.Tensor]], idx: Optional[int] = None):
        """
        recompute internal features

        """
        pass

    @property
    def idx(self) -> int:
        return self._idx

    @idx.setter
    def idx(self, idx: int):
        """ set next internal index modulo neighbourhood length """
        if idx is None:
            idx = 0
        self._idx = idx % self._len

    @property
    def k(self) -> int:
        return self._k


class StaticNeighbourhood(GraphNeighbourhood):
    def __init__(
            self,
            k: int,
            method: str = 'direct',
            grad: bool = False,
            data: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
            length: int = 1,
    ):
        """
        static neighbourhood. calculated only once

        :param k:
        :param method: implementation of neighbourhood computation ['cluster', 'direct']
        :param grad: pass gradients through KNN calculations
        :param data: BxDxN features
        :param length: length of internal state. overridden by data of type list[torch.Tensor]
        """
        super(StaticNeighbourhood, self).__init__(k, method=method, grad=grad)
        self._idx = 0
        self._len = length
        self._idx_warning = True

        if data is not None:
            self._neighbours = []
            if isinstance(data, torch.Tensor):
                self._neighbours.append(self.knn(data))
            elif isinstance(data, list):
                for element in data:
                    self._neighbours.append(self.knn(element))

            else:
                raise RuntimeError('unknown input type for kwarg \'data\'')

        else:
            self._neighbours = [None for _ in range(length)]

        self._len = len(self._neighbours)

    def __call__(self, *args, **kwargs):
        """
        neighbourhood is static, returns the currently stored neighbourhood
        it's the users responsibility to update this for each batch

        :param args: for call compatibility
        :return:
        """
        idx = None
        if 'idx' in kwargs:
            idx = kwargs['idx']

        if idx is None:
            idx = self._idx

        return self._neighbours[idx]

    def recompute(self, data: Union[torch.Tensor, List[torch.Tensor]], idx: int = None):
        if idx is None:
            idx = self._idx

        if isinstance(data, torch.Tensor):
            self._neighbours[idx % self._len] = self.knn(data)

        elif isinstance(data, (list, tuple)):
            for element_idx, element in enumerate(data):
                self._neighbours[(idx + element_idx) % self._len] = self.knn(element)

        else:
            raise RuntimeError('unknown input type for arg \'data\'')
