"""
neighbourhood functions for graph networks

Ludwig Mohr
ludwig.mohr@icg.tugraz.at

"""
import torch
from torch_cluster import knn, radius
from abc import ABC, abstractmethod
from contextlib import nullcontext

from typing import Optional, Union, List


################################################################################
class GraphNeighbourhood(ABC):
    def __init__(
            self,
            k: int,
            r: float = None,
            m: int = None,
            method: str = 'direct',
            grad: bool = False,
            order: str = 'bdn',
            **kwargs
    ):
        """
        return the indices of k nearest neighbours

        for radius based methods:
        up to k points within radius will be returned at random (depending on point order and algorithm internals)
        for method='radius-cluster', up to m points will be queried and a subset of k points will be returned at random,
        if applicable.
        for method='radius', the first points within radius, up to a total number of k, will be returned. the meaning of
        'first' is dependent on point ordering and search algorithm internals.

        :param k: maximum number of neighbours to return
        :param r: search radius for neighbours (for radius search methods)
        :param m: maximum number of neighbours to query for in radius search (only for radius-cluster search method)
        :param method: ['direct', 'cluster', 'radius', 'radius-cluster']
            cluster: use torch cluster knn search
            direct: use direct implementation via greedy neighbourhood search
            radius: hybrid radius/knn search, based on direct implementation
            radius-cluster: hybrid radius/knn search, based on torch_cluster.radius
        :param grad: pass gradients through KNN calculations
        :param order: point order. Batch x Channels x Points ('bdn') or Batch x Points x Channels ('bnd')
        :param kwargs:
        """
        self._k = k
        self._m = m if (m is not None and m > k) else k
        self._r = r
        self._context = nullcontext() if grad else torch.no_grad()

        self._radius_squared = method in ['radius']     # irrelevant for pure kNN,
        if method == 'cluster':
            self.knn = self._knn_cluster

        elif method == 'direct':
            self.knn = self._knn_direct

        elif method == 'radius':
            if not isinstance(self._r, float):
                raise RuntimeError(f'radius parameter must be of type \'float\'')

            self._r = self._r ** 2.0
            self.knn = self._radius

        elif method == 'radius-cluster':
            if not isinstance(self._r, float):
                raise RuntimeError(f'radius parameter must be of type \'float\'')
            self.knn = self._radius_cluster

        else:
            raise RuntimeError(f'unknown knn algorithm: {method}')

        self._idx = 0
        self._len = 1
        self._bdn = order == 'bdn'

    def _knn_cluster(self, features: torch.Tensor):
        """
        calculate the k nearest neighbours for each feature

        :param features: BxDxN features
        :return:
        """
        with self._context:
            if self._bdn:
                features = features.transpose(1, 2)
            features = features.contiguous()
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
            if self._bdn:
                pairwise_inner = -2 * torch.matmul(features.transpose(1, 2), features)
                features_2 = torch.einsum('bdn,bdn->bn', features, features)
            else:
                pairwise_inner = -2 * torch.matmul(features, features.transpose(1, 2))
                features_2 = torch.einsum('bnd,bnd->bn', features, features)

            distance = features_2.unsqueeze(1) + pairwise_inner + features_2.unsqueeze(2)

            _, index = distance.topk(k=self._k + 1, largest=False, dim=2)
            index = index[..., 1:].contiguous()

        return index

    def _radius_cluster(self, features: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        hybrid radius/knn search based on torch_cluster.radius()

        :param features: B x D x N feature vector
        :return:
        """

        with self._context:
            if self._bdn:
                features = features.transpose(1, 2)
            features = features.contiguous()
            batch_size, num_points, feature_dim = features.shape
            device = features.device
            batch = torch.arange(batch_size, dtype=torch.long, device=device).unsqueeze(1).expand(-1, num_points).flatten()

            # 'for each point in y all points in x within a distance of r'
            # radius(x, y, r, b_x, b_y, m, workers, [batch_size])
            index = radius(features.view(-1, feature_dim), features.view(-1, feature_dim), self._r,
                           batch_x=batch.flatten(), batch_y=batch.flatten(),
                           max_num_neighbors=self._m+1)
            # number of neighbours for each query point
            num_neighbours = (index[0, 1:] - index[0, :-1]).nonzero()
            num_neighbours = torch.cat(
                (num_neighbours, torch.tensor((index.shape[1],), dtype=torch.long, device=device).view(1, 1)), 0)
            num_neighbours[1:] = num_neighbours[1:] - num_neighbours[:-1]
            num_neighbours[0] += 1      # correct number for first element
            num_neighbours[-1] -= 1     # correct number for last element

            # if the number of returned neighbours is smaller than the number of queried neighbours
            # return a random sub-sample where applicable
            if self._k < self._m:
                # randomize neighbour order for all but last neighbour (self reference)
                offset = torch.cat(
                    [torch.cat(
                        (
                            x.to('cpu').view(1,) - 1,
                            torch.randperm(x.item()-1, dtype=torch.long, device=device),
                        ),
                        0
                    ) for x in num_neighbours], 0)

            else:
                offset = torch.cat([torch.arange(0, x, dtype=torch.long) for x in num_neighbours], 0)
            put_index = index[0, :] * (self._m+1) + offset
            neighbours = torch.zeros((batch_size * num_points * (self._m+1),), dtype=torch.long, device=device)
            valid = torch.zeros_like(neighbours, dtype=torch.bool)
            neighbours[put_index] = index[1, :]
            valid[put_index] = True

            # reshape to batch_size * num_points x num_neighbours and discard first neighbour (the query point itself)
            neighbours = neighbours.view(batch_size, num_points, self._m+1)[:, :, 1:self._k+1].contiguous()
            valid = valid.view(batch_size, num_points, self._m+1)[:, :, 1:self._k+1].contiguous()

            # shift batch indices back from linear to batched
            neighbours -= torch.arange(0, batch_size * num_points, num_points, device=device
                                       ).reshape(-1, 1, 1).expand(-1, num_points, self._k)

        return neighbours, valid

    def _radius(self, features: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        hybrid radius/knn search based on direct/greedy knn search

        :param features:
        :return:
        """
        with self._context:
            if self._bdn:
                pairwise_inner = -2 * torch.matmul(features.transpose(1, 2), features)
                features_2 = torch.einsum('bdn,bdn->bn', features, features)
            else:
                pairwise_inner = -2 * torch.matmul(features, features.transpose(1, 2))
                features_2 = torch.einsum('bnd,bnd->bn', features, features)

            distance = features_2.unsqueeze(1) + pairwise_inner + features_2.unsqueeze(2)
            trace = torch.arange(distance.shape[1], device=features.device)
            in_range = distance < self._r
            in_range[:, trace, trace] = False

            valid, index = in_range.to(torch.float32).topk(k=self._k, dim=2, sorted=False)  # B x N x K

        return index, valid.to(torch.bool)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        return the current nearest neighbour indices

        :return: BxNxK torch.long tensor
        """
        pass

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

    @property
    def radius(self) -> float:
        return self._r ** 0.5 if self._radius_squared else self._r

    @radius.setter
    def radius(self, r: float):
        if self._radius_squared:
            r = r ** 2.0

        self._r = r


class StaticNeighbourhood(GraphNeighbourhood):
    def __init__(self, *args, length: int = 1, **kwargs):
        """
        static neighbourhood. calculated only once

        :param k:
        :param method: implementation of neighbourhood computation ['cluster', 'direct']
        :param grad: pass gradients through KNN calculations
        :param data: BxDxN features
        :param length: length of internal state. overridden by data of type list[torch.Tensor]
        """
        super(StaticNeighbourhood, self).__init__(*args, **kwargs)
        self._idx = 0
        self._len = length
        self._idx_warning = True

        self._neighbours = [None for _ in range(length)]

        self._len = len(self._neighbours)

    def __call__(self, *args, **kwargs):
        """
        neighbourhood is static, returns the currently stored neighbourhood
        it's the users responsibility to update this for each batch

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
