"""
data loader for ModelNet40


Ludwig Mohr

ludwig.mohr@icg.tugraz.at

"""

import h5py
import numpy as np
import open3d as o3d
from os.path import join, isfile, isdir
from os import listdir
from logging import getLogger

from torch.utils.data import Dataset

from utils.distance import CorrespondenceSearchNN
from utils.misc import merge_dict
from utils.normals import EstimateNormals


################################################################################
class ModelNet40(Dataset):
    """
        data loader that performs data augmentation on the fly
        if the source is a single string (i.e. path), traverse it and expect to sample meshes (raw ModelNet)
        if the source is a dictionary or a single file, expects a pre-sampled clean archive

        for training of a feature networks setting config['model_repetition'] > 1 does not make much sense,
        it's more sensible to increase the number of epochs

    """
    _default_config = {
        'source_randomize': True,
        'target_randomize': True,
        'target_resample': False,
        'min_correspondence': 0.0,
        'prune_ratio': 0.7,
        'sets': {},
        'noise': {
            'source': False,
            'target': False,
            'target_is_source': True,
            'estimate_normals': False,  # re-estimate normals when using additive noise
            'radius': 0.1,              # maximum distance of neighbours for normals estimation
            'neighbours': 30,           # maximum number of neighbours for normals estimation
        },
        'correspondence': {
            'type': 'tree',
            'distance': 0.01,
            'method': 'mutual',
            'iters': 1,
        },
        'normals': False,
        'random_shard': False,
    }

    def __init__(
            self,
            config: dict,
            partition: str,
            start_id: int = 0,
            max_classes: int = 40,
            max_try: int = 100,
            samples: list = None,
            samples_complement: bool = True,
    ):
        """
        :param config   : config dictionary/json (dataset section only)
        :param partition: dataset partition (usually 'test' or 'train')
        :param start_id : starting id for numbering the individual samples
        :param max_classes: maximum number of classes to consider when restricting dataset
        :param max_try: maximum tries to find the required number of correspondences
        :param samples: which samples to take. if training/validation set come from same data source
        :param samples_complement: if complement of samples list is to be taken

        __init__()  :
            determine which loader (from file/sampling or sampled from archive)
            determine length of examples (number of files on disk/length of dataset)

        """
        self._logger = getLogger(__name__)

        # if there are specific values for the partition merge them into standard config
        self._config = merge_dict(self._default_config, config)
        if partition in self._config['sets']:
            self._config = merge_dict(self._config, config['sets'][partition])

        self._partition = partition
        self._start_id = start_id

        self._data = np.zeros((0,))
        self._label = np.zeros((0,))
        self._classes = None

        self._max_try = max_try

        self._prune = False

        self._config['noise']['estimate_normals'] = (
                self._config['normals'] and self._config['noise']['estimate_normals'])
        self._normals = EstimateNormals(
            knn=self._config['noise']['neighbours'], distance=self._config['noise']['radius'])

        if "points" in self._config:
            self._logger.info(f'using dataset \"{partition.upper()}\" with {self._config["points"]} points')
            self._set_prune()
        else:
            self._config['points'] = None

        if self._config['target_resample'] or not self._config['noise']['target_is_source']:
            self._correspondence_search = CorrespondenceSearchNN(**self._config['correspondence'])
            if self._config['min_correspondence'] > 1.:
                self._logger.warning('minimum correspondence ratio must be in range [0., 1.)')
                self._config['min_correspondence'] = 1.

            if self._config['points'] is not None:
                self._min_correspondence = int(self._config['points'] * self._config['min_correspondence'])

                if self._prune:
                    self._min_correspondence = int(self._min_correspondence * self._config['prune_ratio'])

        if 'classes' in self._config:
            if isinstance(self._config['classes'], str):
                # expect a string like ':20', '13:27' or '21:'
                start, stop = self._config['classes'].split(':')
                start = 0 if start == '' else int(start)
                stop = max_classes if stop == '' else int(stop)

                if start >= stop:
                    self._logger.error('start of class range must lie below end of class range')
                    self._classes = None

                else:
                    self._classes = [x for x in range(start, stop)]

            elif isinstance(self._config['classes'], np.ndarray):
                self._classes = self._config['classes']

            elif isinstance(self._config['classes'], list):
                self._classes = np.array(self._config['classes'])

            else:
                self._logger.error(f'can\'t handle classes information of type {str(type(self._config["classes"]))}')
                self._classes = None

            if self._classes is not None:
                self._logger.info(f'using classes: {self._classes}')

        self._do_augment = (self._config['noise']['source'] or self._config['noise']['target']) or \
                           ('sampling' in self._config and (
                                    self._config['sampling']['limits_rotation'] != 0. or
                                    self._config['sampling']['limits_translation'] != 0.))

        # determine loading operation
        if 'archives' in self._config:
            # read archives
            self._get_archives()
            self._get_element = self._get_single_pcd

        else:
            # folder traversal, read meshes from disk
            # dataset root path in config['source_path']
            self._get_meshes()
            self._get_element = self._get_mesh_from_disk

        # reduce training set to allow for a validation set to come from the same source
        if 'shard' in self._config or samples is not None:
            # reduce dataset to random shard or specified items
            num_samples = len(self._data)

            if samples is not None:
                if num_samples <= max(samples):
                    raise RuntimeError(f'sample indices larger than dataset size ({num_samples}) encountered')

                if isinstance(samples, list):
                    samples = np.array(samples)

                if samples_complement:
                    # get complement of samples
                    samples_indices = np.ones((num_samples,), dtype=np.bool)
                    samples_indices[samples] = False
                    self._items = np.arange(num_samples)[samples_indices]

            else:
                # get item indices randomly for desired portion of dataset
                if self._config['shard'] > 1.:
                    self._logger.error(f'invalid entry for \"shard\" in config ({self._config["shard"]}), ignoring')
                    self._config['shard'] = 1.

                elif self._config['random_shard']:
                    # random self._config['shard']% of elements
                    self._items = np.random.choice(
                        num_samples,
                        size=int(num_samples * self._config['shard']),
                        replace=False
                    )
                else:
                    # first self._config['shard']% of elements
                    self._items = np.arange(int(num_samples * self._config['shard']))

            # do the sampling
            if isinstance(self._data, list):
                self._data = [self._data[x] for x in self._items]
                self._label = [self._label[x] for x in self._items]

            else:
                # numpy array
                self._data = self._data[self._items]
                self._label = self._label[self._items]

        else:
            self._items = np.arange(len(self._data))

        if 'samples' in self._config:
            self._logger.info(
                f'reducing dataset \"{self._partition.upper()}\" to {self._config["samples"]} samples')
            self._data = self._data[:self._config['samples']]
            self._label = self._label[:self._config['samples']]
            self._items = self._items[:self._config['samples']]

    def _set_prune(self):
        if 'prune' in self._config:
            self._prune = self._config['prune']

        if self._prune:
            self._prune_points = int(self._config['points'] * self._config['prune_ratio'])
            self._logger.info(f'randomly pruning point clouds to {self._prune_points} points along random plane')

    def _get_meshes(self):
        """
        traverse folder structure, get list of files and classes

        sets:
            self._data (list of _full_ paths to mesh files)
            self._label label associated with corresponding item in self._data

        """
        folders = []
        for class_folder in listdir(self._config['path']):
            class_folder = join(self._config['path'], class_folder, self._partition)

            if isdir(class_folder):
                folders.append(class_folder)

        folders.sort()

        if self._classes is not None:
            folders = [folders[x] for x in self._classes]

        else:
            self._classes = np.arange(len(folders))

        self._data = []
        self._label = []

        file_type = self._config['file_type'].lower() if 'file_type' in self._config else '.off'

        for class_idx, class_folder in zip(self._classes, folders):
            files = [x for x in listdir(class_folder) if x.lower().endswith(file_type)]
            files.sort()
            for cur_file in files:
                file_path = join(class_folder, cur_file)
                if isfile(file_path):
                    self._data.append(file_path)
                    self._label.append(class_idx)

    def _get_mesh_from_disk(self, idx: int):
        """
        load a single mesh

        :param idx:
        :return:
        """
        mesh = o3d.io.read_triangle_mesh(self._data[idx])
        if self._config['normals']:
            mesh.compute_vertex_normals()
        pcd = mesh.sample_points_poisson_disk(
            number_of_points=self._config['points'],
            init_factor=5,
        )

        if self._config['normals']:
            pcd = np.hstack((np.asarray(pcd.points), np.asarray(pcd.normals))).astype('float32')
        else:
            pcd = np.asarray(pcd.points).astype('float32')

        if self._config['source_randomize']:
            np.random.shuffle(pcd)

        return pcd

    def _get_archives(self):
        """
        load data archives
        :return:
        """
        if 'path' not in self._config['archives']:
            raise RuntimeError('config invalid: option \"path\" missing from section \"archives\"')

        if 'sets' not in self._config['archives'] or self._partition not in self._config['archives']['sets']:
            raise RuntimeError(
                f'config invalid: partition \"{self._partition}\" missing from section \"archives:sets\"')

        path = self._config['archives']['path']
        data = []
        label = []

        archives = self._config['archives']['sets'][self._partition]
        if isinstance(archives, str):
            archives = [archives]

        for arch in archives:
            arch_path = join(path, arch)
            if not isfile(arch_path):
                self._logger.warning(f"AugmentationLoader: archive does not exist \"{arch_path}\"")
                continue

            with h5py.File(arch_path, 'r') as f:
                if 'source' in f:
                    cur_data = f['source'][:].astype('float32')
                    cur_label = f['class'][:].astype('int64')

                elif 'data' in f:
                    cur_data = f['data'][:].astype('float32')
                    cur_label = f['label'][:].astype('int64')

                    if self._config['normals']:
                        # load normals and concatenate to data vectors
                        cur_data = np.dstack((cur_data, f['normal'][:].astype('float32')))

                else:
                    self._logger.warning(
                        f"ModelNet40: don't know how to handle archive \"{arch_path}\", skipping")

                # point cloud length info and dependent settings, if not yet available
                if 'points' not in self._config or self._config['points'] is None:
                    self._config['points'] = cur_data.shape[1]
                    self._logger.info(f'point cloud size: {self._config["points"]}')
                    self._set_prune()

                    if self._config['target_resample']:
                        self._min_correspondence = int(self._config['points'] * self._config['min_correspondence'])
                        if self._prune:
                            self._min_correspondence = int(self._min_correspondence * self._config['prune_ratio'])

                if self._classes is not None:
                    keep_idx = np.isin(cur_label, self._classes).flatten()
                    cur_data = cur_data[keep_idx]
                    cur_label = cur_label[keep_idx]

                data.append(cur_data)
                label.append(cur_label)

        if len(data) == 0:
            raise RuntimeError('ModelNet40: no valid archives found')

        self._data = np.concatenate(data, axis=0)
        self._label = np.concatenate(label, axis=0)

    def _get_single_pcd(self, item) -> np.ndarray:
        return self._data[item]

    def __getitem__(self, item: int):
        """
        returns an example for registration consisting of:
            source point cloud
            target point cloud
            rotation source->target
            translation source->target
            class label
            model id (multiple noise variations have the same id)
            full and clean original point cloud (no sub-sampling, no noise)

        :param item:
        :return:
        """
        idx = self._idx_from_item(item)
        pcd_clean = self._get_element(idx)

        reestablish_correspondences = False

        # subsample point cloud
        source_indices = np.arange(pcd_clean.shape[0])
        if self._config['source_randomize']:
            np.random.shuffle(source_indices)

        pcd_source = pcd_clean[source_indices[:min(self._config['points'], pcd_clean.shape[0])], :]

        # randomly generate rotation and translation
        rot, t = self.random_rt()

        source_noise = 0.
        if self._config['noise']['source']:
            source_noise = np.random.normal(
                scale=self._config['noise']['source_variance'],
                size=(self._config['points'], 3),
            ).astype(pcd_source.dtype)

            if 'clip' in self._config['noise'] and self._config['noise']['clip']:
                clip_l = self._config['noise']['clip_range'][0]
                clip_h = self._config['noise']['clip_range'][1]
                source_noise[source_noise > clip_h] = clip_h
                source_noise[source_noise < clip_l] = clip_l

            pcd_source[:, :3] += source_noise

            if self._config['noise']['estimate_normals']:
                # old open3d version: 0.15.2
                pcd_source[:, 3:6] = self._normals(pcd_source)

        # do random pruning on the source point cloud
        if self._prune:
            reestablish_correspondences = True
            pcd_source = self._prune_point_cloud(pcd_source)

        # resample target points? no 1:1 correspondences anymore
        if self._config['target_resample']:
            reestablish_correspondences = True
            indices = np.arange(pcd_clean.shape[0])
            np.random.shuffle(indices)
            pcd_target = pcd_clean[indices[:min(self._config['points'], pcd_clean.shape[0])], :]

        else:
            pcd_target = pcd_clean[source_indices[:min(self._config['points'], pcd_clean.shape[0])], :]

        if self._config['noise']['target']:
            if self._config['noise']['target_is_source'] and self._config['noise']['source']:
                # both have the same noise
                pcd_target[:, :3] += source_noise

            else:
                reestablish_correspondences = True
                target_noise = np.random.normal(
                    scale=self._config['noise']['target_variance'] if 'target_variance' in self._config['noise']
                    else self._config['noise']['source_variance'],
                    size=(self._config['points'], 3),
                ).astype(pcd_target.dtype)

                if 'clip' in self._config['noise'] and self._config['noise']['clip']:
                    clip_l = self._config['noise']['clip_range'][0]
                    clip_h = self._config['noise']['clip_range'][1]
                    target_noise[target_noise > clip_h] = clip_h
                    target_noise[target_noise < clip_l] = clip_l

                pcd_target[:, :3] += target_noise

            if self._config['noise']['estimate_normals']:
                pcd_target[:, 3:6] = self._normals(pcd_target)

        # do random pruning on target point clouds independently
        if self._prune:
            pcd_target = self._prune_point_cloud(pcd_target)

        if reestablish_correspondences:
            # re-establish correspondences
            correspondence = self._correspondence_search.search(
                pcd_target[:, :3],
                pcd_source[:, :3],
            )

        else:
            correspondence = np.arange(0, self._config['points'], dtype=np.int32)

            if self._config['target_randomize']:
                np.random.shuffle(correspondence)
                pcd_target = pcd_target[correspondence]

        # remove canonical orientation of examples
        if not self._config['sampling']['canonical']:
            rot_uncanon, _ = self.random_rt(r_lim=np.pi)

            pcd_source[:, :3] = pcd_source[:, :3].dot(rot_uncanon)
            pcd_target[:, :3] = pcd_target[:, :3].dot(rot_uncanon)
            pcd_clean[:, :3] = pcd_clean[:, :3].dot(rot_uncanon)

            if self._config['normals']:
                pcd_source[:, 3:] = pcd_source[:, 3:].dot(rot_uncanon)
                pcd_target[:, 3:] = pcd_target[:, 3:].dot(rot_uncanon)
                pcd_clean[:, 3:] = pcd_clean[:, 3:].dot(rot_uncanon)

        # if anything, the target is in the canonical orientation, not the source
        # target = source * rot + t
        # target = source.dot(rot.T) + t
        pcd_source[:, :3] = (pcd_source[:, :3] - t).dot(rot)
        if self._config['normals']:
            pcd_source[:, 3:] = pcd_source[:, 3:].dot(rot)

        # return example
        return pcd_source, pcd_target, correspondence, rot, t, self._label[idx], idx + self._start_id, pcd_clean

    def _prune_point_cloud(self, points):
        plane = np.random.random((3,)) - 0.5
        plane /= np.linalg.norm(plane)

        points_prune_idx = np.sort(points[:, :3].dot(plane).argsort()[:self._prune_points])

        return points[points_prune_idx]

    def _idx_from_item(self, item):
        if 'model_repetition' in self._config:
            idx = np.int(np.floor(np.float(item) / np.float(self._config['model_repetition'])))
            return idx

        return item

    def _normals_o3d(self, pcd: np.ndarray) -> np.ndarray:
        """ estimate normals using o3d.PointCloud() """
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(pcd[:, :3])
        if pcd.shape[1] == 6:
            o3d_pcd.normals = o3d.utility.Vector3dVector(pcd[:, 3:6])
        o3d_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=self._config['noise']['radius'], max_nn=self._config['noise']['neighbours']))

        return np.asarray(o3d_pcd.normals)

    def __len__(self):
        if 'model_repetition' in self._config:
            return len(self._data) * self._config['model_repetition']

        return len(self._data)

    def random_rt(
            self,
            r_lim: float = None,
            t_lim: float = None,
            dtype=np.float32,
    ):
        """
        returns a random rotation matrix R in SO(3) and a random translation vector
        t in R3

        the rotation by R is at most r_lim radians
        each element in t is within the range -t_lim <= t <= t_lim

        values are drawn from a uniform distribution

        """
        if r_lim is None:
            r_lim = self._config['sampling']['limits_rotation']
        if t_lim is None:
            t_lim = self._config['sampling']['limits_translation']

        t = (np.random.random(size=(3,)) - .5) * 2. * t_lim

        # create random rotation matrix
        # axis of rotation
        rot = np.eye(3)
        if r_lim > 0.:
            r_ax = np.random.random((3,)) - .5
            r_ax = r_ax / np.linalg.norm(r_ax)

            r_an = (np.random.random() - .5) * 2. * r_lim

            # create R with rodriguez formula
            k = np.array([
                [0., -r_ax[2], r_ax[1]],
                [r_ax[2], 0., -r_ax[0]],
                [-r_ax[1], r_ax[0], 0.],
            ])

            rot = rot + np.sin(r_an) * k + (1. - np.cos(r_an)) * k.dot(k)

        return rot.astype(dtype), t.astype(dtype)

    def unique(self):
        return len(self._data)

    @property
    def items(self):
        return self._items

    @property
    def config(self) -> dict:
        """ hand back config with default settings """
        if hasattr(self, '_correspondence_search'):
            config = {**self._config, 'correspondence': self._correspondence_search.config}
        else:
            config = {**self._config}

        return config
