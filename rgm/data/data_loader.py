#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import glob
import h5py
import torch
import numpy as np
from typing import List
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision

from rgm.utils.build_graphs import build_graphs
import rgm.data.data_transform_syndata as Transforms
# import utils.data_transform4 as Transforms

from rgm.utils.config import cfg

import open3d


# Part of the code is referred from: https://github.com/charlesq34/pointnet

def download():
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www+' --no-check-certificate', zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    # download()
    # DATA_DIR = os.path.dirname(os.path.abspath(__file__))
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(cfg.DATASET_PATH, 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name, 'r')
        # data = f['data'][:].astype('float32')
        data = np.concatenate([f['data'][:], f['normal'][:]], axis=-1)
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def get_transforms(partition: str, num_points: int = 1024,
                   noise_type: str = 'clean', rot_mag: float = 45.0,
                   trans_mag: float = 0.5, partial_p_keep: List = None,
                   non_deterministic_test: bool = False):
    """Get the list of transformation to be used for training or evaluating RegNet

    Args:
        noise_type: Either 'clean', 'jitter', 'crop'.
          Depending on the option, some of the subsequent arguments may be ignored.
        rot_mag: Magnitude of rotation perturbation to apply to source, in degrees.
          Default: 45.0 (same as Deep Closest Point)
        trans_mag: Magnitude of translation perturbation to apply to source.
          Default: 0.5 (same as Deep Closest Point)
        num_points: Number of points to uniformly resample to.
          Note that this is with respect to the full point cloud. The number of
          points will be proportionally less if cropped
        partial_p_keep: Proportion to keep during cropping, [src_p, ref_p]
          Default: [0.7, 0.7], i.e. Crop both source and reference to ~70%
        non_deterministic_test: deactivate non_deterministic testing (for repeatabel results set random state externally)

    Returns:
        train_transforms, test_transforms: Both contain list of transformations to be applied
    """

    partial_p_keep = partial_p_keep if partial_p_keep is not None else [0.7, 0.7]

    if noise_type == "clean":
        # 1-1 correspondence for each point (resample first before splitting), no noise
        if partition == 'train':
            transforms = [Transforms.Resampler(num_points),
                          Transforms.SplitSourceRef(),
                          Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                          Transforms.ShufflePoints()]
        elif non_deterministic_test:
            transforms = [Transforms.Resampler(num_points),
                          Transforms.SplitSourceRef(),
                          Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                          Transforms.ShufflePoints()]
        else:
            transforms = [Transforms.SetDeterministic(),
                          Transforms.FixedResampler(num_points),
                          Transforms.SplitSourceRef(),
                          Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                          Transforms.ShufflePoints()]

    elif noise_type == "jitter":
        # Points randomly sampled (might not have perfect correspondence), gaussian noise to position
        if partition == 'train':
            transforms = [Transforms.SetJitterFlag(),
                          Transforms.SplitSourceRef(),
                          Transforms.Resampler(num_points),
                          Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                          Transforms.RandomJitter(),
                          Transforms.ShufflePoints()]
        elif non_deterministic_test:
            transforms = [Transforms.SetJitterFlag(),
                          Transforms.SplitSourceRef(),
                          Transforms.Resampler(num_points),
                          Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                          Transforms.RandomJitter(),
                          Transforms.ShufflePoints()]
        else:
            transforms = [Transforms.SetJitterFlag(),
                          Transforms.SetDeterministic(),
                          Transforms.SplitSourceRef(),
                          Transforms.Resampler(num_points),
                          Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                          Transforms.RandomJitter(),
                          Transforms.ShufflePoints()]

    elif noise_type == "crop":
        # Both source and reference point clouds cropped, plus same noise in "jitter"
        if partition == 'train':
            transforms = [Transforms.SetCorpFlag(),
                          Transforms.SplitSourceRef(),
                          Transforms.Resampler(num_points),
                          Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                          Transforms.RandomCrop(partial_p_keep),
                          Transforms.RandomJitter(),
                          Transforms.ShufflePoints()]
        elif non_deterministic_test:
            transforms = [Transforms.SetCorpFlag(),
                          Transforms.SplitSourceRef(),
                          Transforms.Resampler(num_points),
                          Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                          Transforms.RandomCrop(partial_p_keep),
                          Transforms.RandomJitter(),
                          Transforms.ShufflePoints()]
        else:
            transforms = [Transforms.SetCorpFlag(),
                          Transforms.SetDeterministic(),
                          Transforms.SplitSourceRef(),
                          Transforms.Resampler(num_points),
                          Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                          Transforms.RandomCrop(partial_p_keep),
                          Transforms.RandomJitter(),
                          Transforms.ShufflePoints()]

    elif noise_type == "cropinv":
        # Both source and reference point clouds cropped, plus same noise in "jitter"
        if partition == 'train':
            transforms = [Transforms.SetCorpFlag(),
                          Transforms.SplitSourceRef(),
                          Transforms.Resampler(num_points),
                          Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                          Transforms.RandomCropinv(partial_p_keep),
                          Transforms.RandomJitter(),
                          Transforms.ShufflePoints()]
        elif non_deterministic_test:
            transforms = [Transforms.SetCorpFlag(),
                          Transforms.SplitSourceRef(),
                          Transforms.Resampler(num_points),
                          Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                          Transforms.RandomCropinv(partial_p_keep),
                          Transforms.RandomJitter(),
                          Transforms.ShufflePoints()]
        else:
            transforms = [Transforms.SetCorpFlag(),
                          Transforms.SetDeterministic(),
                          Transforms.SplitSourceRef(),
                          Transforms.Resampler(num_points),
                          Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                          Transforms.RandomCropinv(partial_p_keep),
                          Transforms.RandomJitter(),
                          Transforms.ShufflePoints()]
    else:
        raise NotImplementedError

    return transforms


class ModelNet40(Dataset):
    def __init__(self, partition='train', unseen=False, transform=None, crossval = False, train_part=False, proportion=0.8):
        # data_shape:[B, N, 3]
        self.data, self.label = load_data(partition)
        if unseen and partition=='train' and train_part is False:
            self.data, self.label = load_data('test')
        self.partition = partition
        self.unseen = unseen
        self.label = self.label.squeeze()
        self.transform = transform
        self.crossval = crossval
        self.train_part = train_part
        if self.unseen:
            ######## simulate testing on first 20 categories while training on last 20 categories
            if self.partition == 'test':
                self.data = self.data[self.label>=20]
                self.label = self.label[self.label>=20]
            elif self.partition == 'train':
                if self.train_part:
                    self.data = self.data[self.label<20]
                    self.label = self.label[self.label<20]
                else:
                    self.data = self.data[self.label<20]
                    self.label = self.label[self.label<20]
        else:
            if self.crossval:
                if self.train_part:
                    self.data = self.data[0:int(self.label.shape[0]*proportion)]
                    self.label = self.label[0:int(self.label.shape[0]*proportion)]
                else:
                    self.data = self.data[int(self.label.shape[0]*proportion):-1]
                    self.label = self.label[int(self.label.shape[0]*proportion):-1]

    def __getitem__(self, item):
        sample = {'points': self.data[item, :, :], 'label': self.label[item], 'idx': np.array(item, dtype=np.int32)}

        if self.transform:
            sample = self.transform(sample)
            # if item==139:
            #     np.save(cfg.DATASET.NOISE_TYPE+'sample'+str(item),sample)

        T_ab = sample['transform_gt']
        T_ba = np.concatenate((T_ab[:,:3].T, np.expand_dims(-(T_ab[:,:3].T).dot(T_ab[:,3]), axis=1)), axis=-1)

        n1_gt, n2_gt = sample['perm_mat'].shape
        A1_gt, e1_gt = build_graphs(sample['points_src'], sample['src_inlier'], n1_gt, stg=cfg.PAIR.GT_GRAPH_CONSTRUCT)
        if cfg.PAIR.REF_GRAPH_CONSTRUCT == 'same':
            A2_gt = A1_gt.transpose().contiguous()
            e2_gt= e1_gt
        else:
            A2_gt, e2_gt = build_graphs(sample['points_ref'], sample['ref_inlier'], n2_gt, stg=cfg.PAIR.REF_GRAPH_CONSTRUCT)

        if cfg.DATASET.NOISE_TYPE != 'clean':
            src_o3 = open3d.geometry.PointCloud()
            ref_o3 = open3d.geometry.PointCloud()
            src_o3.points = open3d.utility.Vector3dVector(sample['points_src'][:, :3])
            ref_o3.points = open3d.utility.Vector3dVector(sample['points_ref'][:, :3])
            src_o3.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            ref_o3.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            sample['points_src'][:, 3:6] = src_o3.normals
            sample['points_ref'][:, 3:6] = ref_o3.normals

        ret_dict = {'Ps': [torch.Tensor(x) for x in [sample['points_src'], sample['points_ref']]],
                    'ns': [torch.tensor(x) for x in [n1_gt, n2_gt]],
                    'es': [torch.tensor(x) for x in [e1_gt, e2_gt]],
                    'gt_perm_mat': torch.tensor(sample['perm_mat'].astype('float32')),
                    'As': [torch.Tensor(x) for x in [A1_gt, A2_gt]],
                    'Ts': [torch.Tensor(x) for x in [T_ab.astype('float32'), T_ba.astype('float32')]],
                    'Ins': [torch.Tensor(x) for x in [sample['src_inlier'], sample['ref_inlier']]],
                    'label': torch.tensor(sample['label']),
                    'raw': torch.Tensor(sample['points_raw']),
                    }
        return ret_dict
        # return pointcloud1.astype('float32'), pointcloud2.astype('float32'), \
        #        n1_gt, n2_gt, \
        #        e1_gt, e2_gt, \
        #        perm_mat.astype('float32'), \
        #        G1_gt.astype('float32'), G2_gt.astype('float32'), \
        #        H1_gt.astype('float32'), H2_gt.astype('float32'),\
        #        R_ab.astype('float32'), R_ba.astype('float32'),\
        #        translation_ab.astype('float32'), translation_ba.astype('float32'),\
        #        euler_ab.astype('float32'), euler_ba.astype('float32')

    def __len__(self):
        return self.data.shape[0]


def get_datasets(partition='train', num_points=1024, unseen=False,
                 noise_type="clean" , rot_mag = 45.0, trans_mag = 0.5,
                 partial_p_keep = [0.7, 0.7], crossval = False, train_part=False, non_deterministic_test=False):
    if cfg.DATASET_NAME=='ModelNet40':
        transforms = get_transforms(partition=partition, num_points=num_points , noise_type=noise_type,
                                    rot_mag = rot_mag, trans_mag = trans_mag, partial_p_keep = partial_p_keep,
                                    non_deterministic_test=non_deterministic_test)
        transforms = torchvision.transforms.Compose(transforms)
        datasets = ModelNet40(partition, unseen, transforms, crossval=crossval, train_part=train_part)
    else:
        raise RuntimeError('please input ModelNet40')

    return datasets


def collate_fn(data: list):
    """
    Create mini-batch data2d for training.
    :param data: data2d dict
    :return: mini-batch
    """
    def pad_tensor(inp):
        assert type(inp[0]) == torch.Tensor
        it = iter(inp)
        t = next(it)
        max_shape = list(t.shape)
        while True:
            try:
                t = next(it)
                for i in range(len(max_shape)):
                    max_shape[i] = int(max(max_shape[i], t.shape[i]))
            except StopIteration:
                break
        max_shape = np.array(max_shape)

        padded_ts = []
        for t in inp:
            pad_pattern = np.zeros(2 * len(max_shape), dtype=np.int64)
            pad_pattern[::-2] = max_shape - np.array(t.shape)
            pad_pattern = tuple(pad_pattern.tolist())
            padded_ts.append(F.pad(t, pad_pattern, 'constant', 0))

        return padded_ts

    def stack(inp):
        if type(inp[0]) == list:
            ret = []
            for vs in zip(*inp):
                ret.append(stack(vs))
        elif type(inp[0]) == dict:
            ret = {}
            for kvs in zip(*[x.items() for x in inp]):
                ks, vs = zip(*kvs)
                for k in ks:
                    assert k == ks[0], "Key value mismatch."
                ret[k] = stack(vs)
        elif type(inp[0]) == torch.Tensor:
            new_t = pad_tensor(inp)
            ret = torch.stack(new_t, 0)
        elif type(inp[0]) == np.ndarray:
            new_t = pad_tensor([torch.from_numpy(x) for x in inp])
            ret = torch.stack(new_t, 0)
        elif type(inp[0]) == str:
            ret = inp
        else:
            raise ValueError('Cannot handle type {}'.format(type(inp[0])))
        return ret

    ret = stack(data)

    # compute CPU-intensive matrix K1, K2 here to leverage multi-processing nature of dataloader
    # if 'Gs' in ret and 'Hs' in ret and :
    #     try:
    #         G1_gt, G2_gt = ret['Gs']
    #         H1_gt, H2_gt = ret['Hs']
    #         sparse_dtype = np.float32
    #         K1G = [kronecker_sparse(x, y).astype(sparse_dtype) for x, y in zip(G2_gt, G1_gt)]  # 1 as source graph, 2 as target graph
    #         K1H = [kronecker_sparse(x, y).astype(sparse_dtype) for x, y in zip(H2_gt, H1_gt)]
    #         K1G = CSRMatrix3d(K1G)
    #         K1H = CSRMatrix3d(K1H).transpose()
    #
    #         ret['Ks'] = K1G, K1H #, K1G.transpose(keep_type=True), K1H.transpose(keep_type=True)
    #     except ValueError:
    #         pass

    return ret


def get_dataloader(dataset, shuffle=False):
    return torch.utils.data.DataLoader(dataset, batch_size=cfg.DATASET.BATCH_SIZE,
                                       shuffle=shuffle, num_workers=cfg.DATALOADER_NUM,
                                       collate_fn=collate_fn, pin_memory=False)


if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    for data in train:
        print(len(data))
        break

