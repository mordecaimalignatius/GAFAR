"""
classes to search nearest neighbours between two point clouds


Ludwig Mohr
ludwig.mohr@icg.tugraz.at

"""
import open3d as o3d
import numpy as np
from typing import Union


########################################################################################################################
def remove_multiples(correspondences: np.ndarray, distances: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """ remove multiple values in correspondences. choose remaining values according to smallest distance """
    unique, inverse, counts = np.unique(correspondences[valid], return_inverse=True, return_counts=True)

    multiples = counts > 1
    linear = np.nonzero(valid)[0]
    distances = distances[valid]
    # didn't come up with a better/more efficient solution
    unique_pos = np.nonzero(multiples)[0]
    for idx in unique_pos:
        # value and position in correspondences/distances/valid
        positions = inverse == idx
        linear_idx = linear[positions]
        valid[linear_idx] = False
        valid[linear_idx[distances[positions].argmin()]] = True

    return valid


########################################################################################################################
class CorrespondenceSearchNN(object):
    """
    search correspondences between two point clouds.
    a point is considered a match if it is within distance of the query point

    supports:
        * mutual nearest neighbour (method = 'mutual', iter = 1)
        * iterative mutual nearest neighbour (method = 'mutual', iter > 1)
        * mutual nearest neighbour plus closest remaining points pairs (method = 'closest')

    """
    def __init__(
            self,
            distance: float = None,
            method: str = 'mutual',
            iters: int = 1,
    ):
        """
        :param distance: maximum squared distance between point pairs [float]
        :param method: method for point pair retrieval [str]
        :param iters: number of iterations to run mutual nearest neighbour search [int]
        """
        if distance is None or distance <= 0.0:
            raise RuntimeError(f'invalid distance threshold: {distance}')
        if method not in ['mutual', 'closest']:
            raise RuntimeError(f'unknown method: <{method}>')

        self._distance = distance
        self._method = method
        self._iter = iters

    def search(self, reference: np.ndarray, source: np.ndarray) -> np.ndarray:
        # run one mutual nearest neighbour search anyway
        correspondence = self.search_mutual(reference, source)

        if self._method not in ['mutual', 'closest']:
            raise RuntimeError(f'method invalid: <{self._method}>')

        elif self._method == 'mutual' and self._iter > 1:
            # run multiple iterations of mutual nearest neighbour search
            correspondence_new = correspondence
            reference_unpaired_global = np.arange(reference.shape[0])
            source_unpaired_global = np.arange(source.shape[0])
            reference_paired = correspondence_new >= 0

            for _ in range(self._iter - 1):
                reference_unpaired = np.logical_not(reference_paired)
                reference_unpaired_global = reference_unpaired_global[reference_unpaired]
                source_unpaired = np.ones(source.shape[0], dtype=bool)
                source_unpaired[correspondence_new[reference_paired]] = False
                source_unpaired_global = source_unpaired_global[source_unpaired]
                if source_unpaired_global.shape[0] == 0 or reference_unpaired_global.shape[0] == 0:
                    # no points left in a point set
                    break

                if reference_unpaired.sum() == 0 or source_unpaired.sum() == 0:
                    break

                reference = reference[reference_unpaired]
                source = source[source_unpaired]

                correspondence_new = self.search_mutual(reference, source)
                if (correspondence_new == -1).all():
                    # no more mutual nearest neighbours
                    break

                # update global correspondence
                reference_paired = correspondence_new >= 0
                correspondence[reference_unpaired_global[reference_paired]] = \
                    source_unpaired_global[correspondence_new[reference_paired]]

        elif self._method == 'closest':
            # for remaining points in reference, add the closest point in remaining points of source
            reference_unpaired = correspondence == -1
            source_unpaired = np.ones(source.shape[0], dtype=bool)
            source_unpaired[correspondence[np.logical_not(reference_unpaired)]] = False
            reference_idx = np.arange(reference.shape[0])[reference_unpaired]
            source_idx = np.arange(source.shape[0])[source_unpaired]

            if reference_unpaired.sum() == 0 or source_unpaired.sum() == 0:
                return correspondence
            reference = reference[reference_unpaired]
            source = source[source_unpaired]

            correspondence_new, distance = self.search_tree(reference, source)
            valid = distance < self._distance
            valid = remove_multiples(correspondence_new, distance, valid)
            correspondence[reference_idx[valid]] = source_idx[correspondence_new[valid]]

            # for remaining points in source, add the closest point in remaining points of reference
            reference_unpaired = np.logical_not(valid)
            reference_idx = reference_idx[reference_unpaired]
            source_unpaired = np.ones(source.shape[0], dtype=bool)
            source_unpaired[correspondence_new[valid]] = False
            source_idx = source_idx[source_unpaired]

            if reference_unpaired.sum() == 0 or source_unpaired.sum() == 0:
                return correspondence
            reference = reference[reference_unpaired]
            source = source[source_unpaired]

            correspondence_inv, distance = self.search_tree(source, reference)
            valid = distance < self._distance
            valid = remove_multiples(correspondence_inv, distance, valid)
            correspondence[reference_idx[correspondence_inv[valid]]] = source_idx[valid]

        return correspondence

    def search_mutual(self, reference: np.ndarray, source: np.ndarray) -> np.ndarray:
        """
        for each point in ref, search the closest point in src within maximum distance
        for each point in src, search the closest point in ref within maximum distance

        if they are mutual nearest neighbours, count them as correspondence

        the index array matches points from src to ref: src[idx] -> ref

        :param reference: reference point cloud, Nx3
        :param source: source point cloud, Mx3
        :return:
        """
        correspondence, distance = self.search_tree(reference, source)
        valid = distance < self._distance

        # inverse search
        correspondence_inv, _ = self.search_tree(source, reference)

        # get mutual correspondences only
        mutual = np.logical_and(
            correspondence_inv[correspondence] == np.arange(reference.shape[0]),
            valid,
        )
        correspondence[np.logical_not(mutual)] = -1

        return correspondence

    @staticmethod
    def search_tree(reference: np.ndarray, source: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """ k search nearest neighbours in source. returns neighbour indices and squared distances """
        reference_o3d = o3d.utility.Vector3dVector(reference[:, :3])
        source_o3d = o3d.utility.Vector3dVector(source[:, :3])

        pcd = o3d.geometry.PointCloud()
        pcd.points = source_o3d
        search_tree = o3d.geometry.KDTreeFlann(pcd)

        c_idx = np.full((reference.shape[0], k), fill_value=-1, dtype=int)
        c_dist = np.zeros((reference.shape[0], k))

        for idx, point in enumerate(reference_o3d):
            _, src_idx, dist = search_tree.search_knn_vector_3d(point, k)
            c_idx[idx] = np.asarray(src_idx)
            c_dist[idx] = np.asarray(dist)

        return c_idx.squeeze(), c_dist.squeeze()

    @property
    def config(self) -> dict[str, Union[int, float, str]]:
        return {
            'distance': self._distance,
            'method': self._method,
            'iters': self._iter,
        }
