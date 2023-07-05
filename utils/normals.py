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

    this is a reimplementation of the normal estimation of open3d, since
    open3d.geometry.PointCloud.estimate_normals() dies in torch Dataloader worker processes

    if initial normals are provided, orient normals according to initial vectors.

    for each point the k nearest neighbours within a maximum distance are considered

    """
    def __init__(self, knn: int = 30, distance: Union[float, None] = 0.1):
        self._use_distance = distance is not None
        self._knn = knn + 1
        self._distance = distance if self._use_distance else None

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """ Nx3 (without normals) or Nx6 points (with normals) """

        pcd = o3d.geometry.PointCloud()
        points_o3d = o3d.utility.Vector3dVector(points[:, :3])
        pcd.points = points_o3d
        search_tree = o3d.geometry.KDTreeFlann(pcd)
        closest = np.zeros((points.shape[0], self._knn), dtype=int)

        if self._use_distance:
            weight = np.zeros(closest.shape)
            for i, point in enumerate(points_o3d):
                num, idx, _ = search_tree.search_hybrid_vector_3d(point, self._distance, self._knn)
                closest[i, :num] = np.asarray(idx)
                weight[i, :num] = 1.0
        else:
            weight = np.ones(closest.shape)
            for i, point in enumerate(points_o3d):
                _, idx, _ = search_tree.search_knn_vector_3d(point, self._knn)
                closest[i] = np.asarray(idx)

        # create centered local neighbourhood
        points_nn = np.take_along_axis(points[:, :3][:, None, :], np.tile(closest[..., None], (1, 1, 3)), axis=0)
        points_nn = points_nn * weight[..., None]
        points_nn = points_nn - points_nn.sum(axis=1, keepdims=True) / weight.sum(axis=1).reshape(-1, 1, 1)

        # PCA using covariance matrix
        cov = np.einsum('nkf,nkd->nfd', points_nn, points_nn * weight[..., None])
        cov[weight.sum(axis=1) < 3] = np.eye(3)
        w, v = np.linalg.eig(cov)
        w = np.argmin(w.real, axis=1)

        normals = v.real[np.arange(w.shape[0]), :, w]

        # normalize vectors to unit length
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)

        if points.shape[1] == 6:
            # orient normals
            direction = np.sign(np.einsum('md,md->m', points[:, 3:6], normals))[:, None]
            normals = normals * direction

        return normals
