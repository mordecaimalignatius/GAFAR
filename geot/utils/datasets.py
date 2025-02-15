"""

adapted from GeoTransformer  https://github.com/qinzheng93/GeoTransformer
experiments/*/datasets.py

"""
import torch

from .data import build_dataloader_stack_mode, registration_collate_fn_stack_mode
from .torch import reset_seed_worker_init_fn


def train_valid_data_loader(cfg):
    train_dataset = cfg.data.dataset(
        cfg.data.dataset_root,
        'train',
        point_limit=cfg.train.point_limit,
        use_augmentation=cfg.train.use_augmentation,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_rotation=cfg.train.augmentation_rotation,
    )
    train_loader = build_dataloader_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        [],
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        precompute_data=False,
        shuffle=True,
        distributed=False,
    )

    valid_dataset = cfg.data.dataset(
        cfg.data.dataset_root,
        'val',
        point_limit=cfg.test.point_limit,
        use_augmentation=False,
    )
    valid_loader = build_dataloader_stack_mode(
        valid_dataset,
        registration_collate_fn_stack_mode,
        0, 0.0, 0.0, [],
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        precompute_data=False,
        shuffle=False,
        distributed=False,
    )

    return train_loader, valid_loader


def test_data_loader(cfg, benchmark: str):
    test_dataset = cfg.data.dataset(
        cfg.data.dataset_root,
        benchmark,
        use_augmentation=False,
        point_limit=cfg.test.point_limit,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        sampler=None,
        collate_fn=None,
        worker_init_fn=reset_seed_worker_init_fn,
    )

    return test_loader
