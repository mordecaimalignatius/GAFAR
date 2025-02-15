"""
Kitti Odometry Dataset loader
adapted from GeoTransformer: https://github.com/qinzheng93/GeoTransformer

Ludwig Mohr
ludwig.mohr@icg.tugraz.at

"""

import numpy as np
import torch.utils.data
import pickle

from pathlib import Path
from logging import getLogger
from random import random
from scipy.spatial.transform import Rotation
from copy import deepcopy


from utils.misc import merge_dict
from utils.distance import CorrespondenceSearchNN
from utils.normals import EstimateNormals


########################################################################################################################
def load_pickle(filename: Path):
    """ geotransformer.utils.common """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def get_transform_from_rotation_translation(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    r"""
    geotransformer.utils.pointcloud
    Get rigid transform matrix from rotation matrix and translation vector.

    Args:
        rotation (array): (3, 3)
        translation (array): (3,)

    Returns:
        transform: (4, 4)
    """
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def get_rotation_translation_from_transform(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""
    geotransformer.utils.pointcloud
    Get rotation matrix and translation vector from rigid transform matrix.

    Args:
        transform (array): (4, 4)

    Returns:
        rotation (array): (3, 3)
        translation (array): (3,)
    """
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return rotation, translation


def apply_transform(points: np.ndarray, transform: np.ndarray, normals: np.ndarray = None):
    """ geotransformer.utils.pointcloud """
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    if points.shape[1] == 6:
        # joint normals
        points = np.hstack((np.matmul(points[:, :3], rotation.T) + translation,
                            np.matmul(points[:, 3:6], rotation.T)))
        return points
    points = np.matmul(points, rotation.T) + translation
    if normals is not None:
        normals = np.matmul(normals, rotation.T)
        return points, normals
    else:
        return points


def random_sample_rotation(rotation_factor: float = 1.0) -> np.ndarray:
    """ geotransformer.utils.pointcloud """
    # angle_z, angle_y, angle_x
    euler = np.random.rand(3) * np.pi * 2 / rotation_factor  # (0, 2 * pi / rotation_range)
    rotation = Rotation.from_euler('zyx', euler).as_matrix()
    return rotation


########################################################################################################################
class KittiOdometryDataset(torch.utils.data.Dataset):
    ODOMETRY_KITTI_DATA_SPLIT = {
        'train': ['00', '01', '02', '03', '04', '05'],
        'val': ['06', '07'],
        'test': ['08', '09', '10'],
    }
    _default_config = {
        'sets': {},
        'augment': {
            'use_augmentation': False,
            'noise_scale': 0.01,        # may have noise_soft_start
            'scale_min': 0.8,
            'scale_max': 1.2,
            'rotation': 1.0,            # u(0, 1) * np.pi * 2 / rotation_factor     may have rotation_soft_start
            'shift': 2.0,
            'radius': 4.25,             # maximum distance of neighbours for normals estimation
            'neighbours': 30,           # maximum number of neighbours for normals estimation
            'estimate_normals': True,   # re-estimate (noisy) normals
        },
        'correspondence': {
            'distance': 0.5625,        # 0.75^2
            'method': 'closest',
        },
        "registration_recall": {
            "rotation": 5.0,
            "translation": 2.0
        },
        'normals': True,
        'approx_normals': True,         # approximate normals neighbourhood
    }

    def __init__(
            self,
            config: dict,
            partition: str,
    ):
        super(KittiOdometryDataset, self).__init__()
        self._logger = getLogger(__name__)

        self._config = merge_dict(self._default_config, config)
        if partition in self._config['sets']:
            self._config = merge_dict(self._config, self._config['sets'][partition])

        self.dataset_root = Path(self._config['path'])
        self.subset = partition
        self.point_limit = self._config['points'] if 'points' in self._config else None

        if self.point_limit is not None:
            self._logger.info(f'using split {self.subset.upper()} with {self.point_limit} points')

        self._search = CorrespondenceSearchNN(**self._config['correspondence'])

        # self.metadata = load_pickle((self.dataset_root / 'metadata') / f'{partition}.pkl')
        self.metadata = load_pickle(self.dataset_root / self._config['meta'][partition])

        self._normals = self._config['normals']
        if self._config['augment']['estimate_normals']:
            self._normals_estimator = EstimateNormals(
                knn=self._config['augment']['neighbours'], distance=self._config['augment']['radius'],
                approx=self._config['approx_normals'], randomize=True,
            ) if self._normals else None
        else:
            self._logger.info(f'not re-estimating noisy normals!')
            self._normals_estimator = None

        # augmentation soft starts
        self._rotation = self._config['augment']['rotation']
        self._noise_scale = self._config['augment']['noise_scale']

    def _augment_point_cloud(self, ref_points, src_points, transform, key_ref, key_src):
        rotation, translation = get_rotation_translation_from_transform(transform)
        ref_clean = ref_points.copy()
        src_clean = src_points.copy()
        # add uniform noise to position (as GeoTransformer does)
        ref_points[:, :3] += (np.random.rand(ref_points.shape[0], 3) - 0.5) * self._noise_scale
        src_points[:, :3] += (np.random.rand(src_points.shape[0], 3) - 0.5) * self._noise_scale

        # get sampling indices
        idx_ref = np.random.permutation(ref_points.shape[0])[: min(self.point_limit, ref_points.shape[0])]
        idx_src = np.random.permutation(src_points.shape[0])[: min(self.point_limit, src_points.shape[0])]

        # random rotation
        aug_rotation = random_sample_rotation(self._rotation)
        if random() > 0.5:
            ref_points[:, :3] = np.matmul(ref_points[:, :3], aug_rotation.T)
            if self._normals:
                ref_points[:, 3:6] = np.matmul(ref_points[:, 3:6], aug_rotation.T)
            rotation = np.matmul(aug_rotation, rotation)
            translation = np.matmul(aug_rotation, translation)
            ref_clean[:, :3] = np.matmul(ref_clean[:, :3], aug_rotation.T)
            if self._normals:
                ref_clean[:, 3:6] = np.matmul(ref_clean[:, 3:6], aug_rotation.T)
        else:
            src_points[:, :3] = np.matmul(src_points[:, :3], aug_rotation.T)
            if self._normals:
                src_points[:, 3:6] = np.matmul(src_points[:, 3:6], aug_rotation.T)
            rotation = np.matmul(rotation, aug_rotation.T)
            if src_clean is not None:
                src_clean[:, :3] = np.matmul(src_clean[:, :3], aug_rotation.T)
                if self._normals:
                    src_clean[:, 3:6] = np.matmul(src_clean[:, 3:6], aug_rotation.T)

        # random scaling
        scale = random()
        scale = self._config['augment']['scale_min'] + (
                self._config['augment']['scale_max'] - self._config['augment']['scale_min']) * scale
        ref_points[:, :3] *= scale
        src_points[:, :3] *= scale
        translation *= scale
        ref_clean[:, :3] *= scale
        src_clean[:, :3] *= scale
        # random shift
        ref_shift = np.random.uniform(-self._config['augment']['shift'], self._config['augment']['shift'], 3)
        src_shift = np.random.uniform(-self._config['augment']['shift'], self._config['augment']['shift'], 3)
        ref_points[:, :3] += ref_shift
        src_points[:, :3] += src_shift
        translation = -np.matmul(src_shift[None, :], rotation.T) + translation + ref_shift
        ref_clean[:, :3] += ref_shift
        src_clean[:, :3] += src_shift
        # compose transform from rotation and translation
        transform = get_transform_from_rotation_translation(rotation, translation)

        # re-estimate noisy normals
        if self._normals_estimator is not None:
            ref_points = np.hstack((ref_points[idx_ref, :3],
                                    self._normals_estimator(ref_points, key=key_ref, indices=idx_ref
                                                            ).astype(np.float32).reshape(idx_ref.shape[0], -1)))
            src_points = np.hstack((src_points[idx_src, :3],
                                    self._normals_estimator(src_points, key=key_src, indices=idx_src
                                                            ).astype(np.float32).reshape(idx_src.shape[0], -1)))

            ref_clean = np.hstack((ref_clean[idx_ref, :3],
                                   self._normals_estimator(ref_clean, key=key_ref, indices=idx_ref
                                                           ).astype(np.float32).reshape(idx_ref.shape[0], -1)))
            src_clean = apply_transform(src_clean, transform)
            src_clean = np.hstack((src_clean[idx_src, :3],
                                   self._normals_estimator(src_clean, key=key_src, indices=idx_src
                                                           ).reshape(idx_src.shape[0], -1))).astype(np.float32)

        else:
            ref_points = ref_points[idx_ref, :]
            src_points = src_points[idx_src, :]

            ref_clean = ref_clean[idx_ref, :]
            src_clean = apply_transform(src_clean[idx_src, :], transform)

        return ref_points, src_points, transform, ref_clean, src_clean

    @staticmethod
    def _load_point_cloud(file_name):
        return np.load(file_name)

    def __getitem__(self, index) -> dict[str, np.ndarray]:
        """
        returns an example for registration as a dictionary with keys:
            points_src:     source point cloud
            points_ref:     target point cloud
            points_clean:   noise free, joint source and reference point clouds in coordinate system of points_ref
            rotation:       rotation source->target
            translation:    translation source->target
            label:          class label
            id:             model id (multiple noise variations have the same id)
        """
        metadata = self.metadata[index]

        ref_points = self._load_point_cloud(self.dataset_root / metadata['pcd0'])
        src_points = self._load_point_cloud(self.dataset_root / metadata['pcd1'])
        transform = metadata['transform']

        if not self._normals:
            ref_points = ref_points[:, :3]
            src_points = src_points[:, :3]

        if self._config['augment']['use_augmentation']:
            ref_points, src_points, transform, ref_clean, src_clean = \
                self._augment_point_cloud(ref_points, src_points, transform, metadata['pcd0'], metadata['pcd1'])
        else:
            ref_indices = np.random.permutation(ref_points.shape[0])[: min(self.point_limit, ref_points.shape[0])]
            src_indices = np.random.permutation(src_points.shape[0])[: min(self.point_limit, src_points.shape[0])]
            if self._normals_estimator is not None:
                src_normals = self._normals_estimator(
                    apply_transform(src_points, transform),
                    key=metadata['pcd1'], indices=src_indices).reshape(src_indices.shape[0], -1)
                src_clean = np.hstack((apply_transform(src_points[src_indices, :3], transform), src_normals)
                                      ).astype(np.float32)
                ref_normals = self._normals_estimator(
                    ref_points, key=metadata['pcd0'], indices=ref_indices).reshape(ref_indices.shape[0], -1)
                ref_points = np.hstack((ref_points[ref_indices, :3], ref_normals))
                src_normals = self._normals_estimator(
                    src_points, key=metadata['pcd1'], indices=src_indices).reshape(ref_indices.shape[0], -1)
                src_points = np.hstack((src_points[src_indices, :3], src_normals))
            else:
                ref_points = ref_points[ref_indices, :]
                src_points = src_points[src_indices, :]
                src_clean = apply_transform(src_points, transform)

            ref_clean = ref_points

        correspondence = self._search.search(ref_points[:, :3], apply_transform(src_points[..., :3], transform))

        rot, t = get_rotation_translation_from_transform(transform)

        sample = {
            'points_src': src_points,
            'points_ref': ref_points,
            'correspondence': correspondence,
            'points_clean': np.vstack((ref_clean, src_clean)),
            'rotation': rot.astype(np.float32),
            'translation': t.astype(np.float32),
            'label': metadata['seq_id'],
            'id': index,
        }

        return sample

    def __len__(self):
        return len(self.metadata)

    def get_full(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """ only works for non-augmented point clouds """
        metadata = self.metadata[index]
        reference = np.load(self.dataset_root / metadata['pcd0'])
        source = np.load(self.dataset_root / metadata['pcd1'])

        return source, reference

    @property
    def get_normals_parameter(self) -> dict:
        return {'nn': self._config['augment']['neighbours'], 'radius': self._config['augment']['radius']}

    @property
    def config(self) -> dict:
        """ hand back config with default settings """
        return deepcopy(self._config)

    @property
    def points(self) -> int:
        return self.point_limit

    def normals(self, value: bool = True):
        self._normals = value
