"""
estimate point normals for all points in a point set

Ludwig Mohr
ludwig.mohr@icg.tugraz.at

"""

import numpy as np
import open3d as o3d

from typing import Union


class EstimateNormals(object):
    """
    estimate point normals for each point in a point set by returning the eigenvector of the smallest eigenvalue
    of PCA

    uses open3d KDTree neighbour search

    if initial normals are provided, orient normals according to initial vectors.

    for each point the k nearest neighbours within a maximum distance are considered

    """
    def __init__(self, knn: int = 30, distance: Union[float, None] = 0.1, approx: bool = True,
                 randomize: bool = True):
        self._use_distance = distance is not None
        self._knn = knn + 1
        self._distance = distance if self._use_distance else None
        self._approx = approx
        self._randomize = randomize

        self._idx_mem = {}

    def __call__(self, points: np.ndarray, key: Union[str, int] = None, indices: np.ndarray = None) -> np.ndarray:
        """ Nx3 (without normals) or Nx6 points (with normals) """
        if indices is None:
            indices = np.arange(points.shape[0])

        if key is None or key not in self._idx_mem:
            pcd = o3d.geometry.PointCloud()
            points_o3d = o3d.utility.Vector3dVector(points[:, :3])
            pcd.points = points_o3d
            search_tree = o3d.geometry.KDTreeFlann(pcd)
            closest = np.zeros((points.shape[0], self._knn), dtype=np.int32)

            if self._use_distance:
                num_nn = np.zeros(points.shape[0], dtype=np.int32)
                for i, point in enumerate(points_o3d):
                    num, idx, _ = search_tree.search_hybrid_vector_3d(point, self._distance, self._knn)
                    closest[i, :num] = np.asarray(idx)
                    num_nn[i] = num
            else:
                num_nn = np.full(points.shape[0], dtype=np.int32, fill_value=self._knn)
                for i, point in enumerate(points_o3d):
                    _, idx, _ = search_tree.search_knn_vector_3d(point, self._knn)
                    closest[i] = np.asarray(idx)

            if key is not None and self._approx:
                self._idx_mem[key] = (closest, None if np.all(num_nn == self._knn) else num_nn)

        else:
            closest, num_nn = self._idx_mem[key]

        weight = np.ones((indices.shape[0], self._knn), dtype=np.float32)
        if self._use_distance and num_nn is not None:
            for i, num in enumerate(num_nn[indices]):
                weight[i, num:] = 0.0

        # create centered local neighbourhood
        points_nn = np.take_along_axis(points[:, :3][:, None, :], np.tile(closest[indices, :, None], (1, 1, 3)), axis=0)
        points_nn = points_nn * weight[..., None]
        points_nn = points_nn - points_nn.sum(axis=1, keepdims=True) / weight.sum(axis=1).reshape(-1, 1, 1)

        # PCA using covariance matrix
        cov = np.einsum('nkf,nkd->nfd', points_nn, points_nn * weight[..., None])
        cov[weight.sum(axis=1) < 3] = np.eye(3)
        w, v = np.linalg.eig(cov)
        w = np.argmin(w.real, axis=1)

        normals = v.real[np.arange(w.shape[0]), :, w]

        # # normalize vectors to unit length
        # normals /= np.linalg.norm(normals, axis=1, keepdims=True)

        if self._randomize:
            sign = np.sign(np.random.rand(indices.shape[0]) - 0.5)[:, None].astype(normals.dtype)
            normals = normals * sign

        elif points.shape[1] == 6:
            # orient normals
            sign = np.sign(np.einsum('md,md->m', points[indices, 3:6], normals))[:, None]
            normals = normals * sign

        return normals
