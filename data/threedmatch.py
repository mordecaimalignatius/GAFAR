"""
data loader for 3D Match dataset

adapted from GeoTransformer: https://github.com/qinzheng93/GeoTransformer


Ludwig Mohr
ludwig.mohr@icg.tugraz.at

"""

import torch
import torch.utils.data
import numpy as np
import pickle

from copy import deepcopy
from logging import getLogger
from pathlib import Path


from utils.misc import merge_dict
from utils.distance import CorrespondenceSearchNN
from utils.normals import EstimateNormals


########################################################################################################################
class ThreeDMatch(torch.utils.data.Dataset):
    """
    dataset for 3D Match
        * sampling from fragment pairs to set number of points
        * establish correspondences

        config dict entries:
            path:       str         source path of archives
            archives:   List[str]   archives to load
            points:     int         number of points to (randomly) down sample both source and reference point sets to
            orientation: {
                augment: [False, True]  augment dataset by introducing an additional rotation to both source
                                            and reference point sets
                angle:   radians         maximum angle for orientation augmentation
                }
            transformation: {
                augment: [False, True]  augment dataset by introducing an additional rotation to the source point set
                angle:   radians        maximum angle for transformation augmentation
                }
            reversal: {
                augment: [False, True]      augment dataset by performing role reversal (source <-> reference)
                probability: float [0., 1.] probability with which to perform role reversal
                }

    """
    _default_config = {
        'orientation': {
            'augment': False,
            'angle': 2*np.pi,
        },
        'transformation': {
            'augment': False,
            'angle': np.pi/2,
        },
        'reversal': {
            'augment': False,
            'probability': 0.5,
        },
        'noise': {
            'augment': False,
            'scale': 0.005,
        },
        'correspondence': {
            'distance': 0.01,
            'method': 'closest',
        },
        'overlap_min': 0.3,
        'overlap_max': 1.0,
        'sets': {},
        'normals': True,
    }

    _default_normals = {
        'radius': 0.5,                  # was 1.0 for pre-sampling/normals estimation on reduced point set
        'nn': 30,
        'approx': True,                 # approximate (i.e. save) neighbourhood search for normals estimation
                                        # if not pre-sampling
        'randomize': True,              # randomize normals orientation (flip orientation by random +/- 1)
    }

    def __init__(
            self,
            config: dict,
            partition: str,
    ):
        self._logger = getLogger(__name__)
        self._partition = partition

        # process config
        self._config = merge_dict(self._default_config, config)
        self._normals = None
        if self._config['normals']:
            self._config['estimation'] = merge_dict(
                self._default_normals, self._config['estimation'] if 'estimation' in self._config else {})
            self._normals = EstimateNormals(
                knn=self._config['estimation']['nn'], distance=self._config['estimation']['radius'],
                approx=self._config['estimation']['approx'], randomize=True,
            )

        # if there are specific values for the partition merge them into standard config
        self._config_general = self._config
        if partition in self._config['sets']:
            self._config_general = deepcopy(self._config)
            self._config = merge_dict(self._config, config['sets'][partition])

        self._reduce_point_set = None
        if 'points' in self._config:
            self._logger.info(f'using partition \"{partition.upper()}\" with {self._config["points"]} points')
            self._reduce_point_set = self._config['points']

        with open(Path(self._config['path']) / 'metadata' / f'{partition}.pkl', 'rb') as f:
            self._metadata_list = pickle.load(f)
            if self._config['overlap_min'] is not None:
                self._metadata_list = [
                    x for x in self._metadata_list
                    if self._config['overlap_min'] < x['overlap'] <= self._config['overlap_max']]
        self._path = Path(self._config['path']) / 'data'

        self._return_normals = self._config['normals']

        self._correspondence = CorrespondenceSearchNN(**self._config['correspondence'])

        self._angle = self._config['transformation']['angle']
        self._scale = self._config['noise']['scale']

    def _load_point_cloud(self, file_name):
        return torch.load(self._path / file_name).astype(np.float32)

    @staticmethod
    def _get_rotation(max_angle_rad: float) -> np.ndarray:
        # create random rotation matrix
        rot = np.eye(3, dtype=np.float32)
        if max_angle_rad > 0.:
            # axis of rotation
            rot_axis = np.random.random((3,)) - .5
            rot_axis = rot_axis / np.linalg.norm(rot_axis)

            angle = (np.random.random() - .5) * 2. * max_angle_rad

            # create R with rodriguez formula
            k = np.array([
                [0., -rot_axis[2], rot_axis[1]],
                [rot_axis[2], 0., -rot_axis[0]],
                [-rot_axis[1], rot_axis[0], 0.],
            ])

            rot = rot + np.sin(angle) * k + (1. - np.cos(angle)) * k.dot(k)

        return rot.astype(np.float32)

    def __len__(self) -> int:
        return len(self._metadata_list)

    def __getitem__(self, item: int) -> dict[str, np.ndarray]:
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
        metadata = self._metadata_list[item]
        # reference_id = metadata['frag_id0']
        # source_id = metadata['frag_id1']

        # subsampling of point sets
        points_reference = self._load_point_cloud(metadata['pcd0'])
        points_source = self._load_point_cloud(metadata['pcd1'])

        rotation = metadata['rotation'].astype(np.float32)
        translation = metadata['translation'].astype(np.float32)

        # orientation augmentation
        if self._config['orientation']['augment']:
            rot = self._get_rotation(self._config['orientation']['angle'])
            points_reference = np.matmul(points_reference, rot.T)
            source_rotation = rotation.T @ rot @ rotation
            points_source = np.matmul(points_source, source_rotation.T)
            translation = np.matmul(rot, translation)

        # transformation augmentation
        if self._config['transformation']['augment']:
            rot = self._get_rotation(self._angle)
            points_source = np.matmul(points_source, rot)
            rotation = np.matmul(rotation, rot)

        # role reversal
        if self._config['reversal']['augment'] and np.random.rand() < self._config['reversal']['probability']:
            rotation = rotation.T
            translation = np.matmul(rotation, -translation)
            points_source, points_reference = points_reference, points_source

            key_ref, key_src = metadata['pcd1'], metadata['pcd0']
        else:
            key_ref, key_src = metadata['pcd0'], metadata['pcd1']

        translation = translation.flatten()

        # get sampling indices
        if self._reduce_point_set is not None:
            if points_reference.shape[0] > self._reduce_point_set:
                idx_reference = np.random.permutation(points_reference.shape[0])[: self._reduce_point_set]
            else:
                idx_reference = np.hstack((np.arange(points_reference.shape[0]),
                                           np.random.permutation(points_reference.shape[0])[:(
                                                   self._reduce_point_set - points_reference.shape[0])]))
            if points_source.shape[0] > self._reduce_point_set:
                idx_source = np.random.permutation(points_source.shape[0])[: self._reduce_point_set]
            else:
                idx_source = np.hstack((np.arange(points_source.shape[0]),
                                        np.random.permutation(points_source.shape[0])[:(
                                                self._reduce_point_set - points_source.shape[0])]))
        else:
            idx_reference = np.arange(points_reference.shape[0])
            idx_source = np.arange(points_source.shape[0])

        # normals for clean point cloud
        if self._return_normals:
            points_clean = np.vstack((
                np.hstack((points_reference[idx_reference],
                           self._normals(points_reference, key=key_ref, indices=idx_reference))),
                np.hstack((points_source[idx_source] @ rotation.T + translation,
                           self._normals(points_source, key=key_src, indices=idx_source) @ rotation.T))
            ))
        else:
            points_clean = np.vstack((points_reference[idx_reference],
                                      points_source[idx_source] @ rotation.T + translation))

        # add random uniform noise
        if self._config['noise']['augment']:
            points_reference += ((np.random.rand(points_reference.shape[0], 3) - 0.5) * self._scale).astype(np.float32)
            points_source += ((np.random.rand(points_source.shape[0], 3) - 0.5) * self._scale).astype(np.float32)

        # normals for possibly noisy point cloud
        if self._return_normals:
            points_reference_sampled = np.hstack((points_reference[idx_reference],
                                                  self._normals(points_reference, key=key_ref, indices=idx_reference)))
            points_source_sampled = np.hstack((points_source[idx_source],
                                               self._normals(points_source, key=key_src, indices=idx_source)))
        else:
            points_reference_sampled = points_reference[idx_reference]
            points_source_sampled = points_source[idx_source]

        # correspondences
        correspondences = self._correspondence.search(points_reference_sampled[:, :3],
                                                      points_source_sampled[:, :3] @ rotation.T + translation)

        # create training sample dictionary
        sample = {
            'points_src': points_source_sampled,
            'points_ref': points_reference_sampled,
            'correspondence': correspondences,
            'points_clean': points_clean,
            'rotation': rotation,
            'translation': translation,
            'label': metadata['scene_name'],
            'id': item,
            'overlap': metadata['overlap']
        }

        return sample

    def get_full(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        # return full and raw point clouds
        metadata = self._metadata_list[item]
        reference = torch.load(self._path / metadata['pcd0']).astype(np.float32)
        source = torch.load(self._path / metadata['pcd1']).astype(np.float32)

        return source, reference

    @property
    def get_normals_parameter(self) -> dict:
        return {'nn': self._config['estimation']['nn'], 'radius': self._config['estimation']['radius']}

    @property
    def config(self):
        return self._config_general

    @property
    def config_partition(self):
        return self._config

    @property
    def points(self) -> int:
        return self._reduce_point_set

    def normals(self, value: bool = True):
        self._return_normals = value
