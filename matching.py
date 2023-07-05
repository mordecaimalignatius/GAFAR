"""
training/testing of 3D point matching
(on small point clouds -> maybe also on large ones)

redo/reorganization of matching1024.py, hopefully better

Ludwig Mohr
ludwig.mohr@icg.tugraz.at

"""
import signal
import argparse
import torch
import json

import numpy as np
import utils.sigint

from copy import deepcopy
from shutil import copyfile
from time import time
from logging import Logger, basicConfig
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Union

from gafar import GAFAR, Loss
from gafar.misc import scores_to_prediction, estimate_transform
from utils.metrics import transformation_error, evaluate_performance, chamfer_distance_modified

from data import ModelNet40

from utils.logs import get_logger, get_tensorboard_log_path
from utils.logs import matching_results_to_tensorboard as log_tensorboard
from utils.logs import matching_log_result as log_console
from utils.config import write_json
from utils.misc import merge_dict, get_device
from utils.random_state import worker_random_init, rng_save_restore


################################################################################
_default_config = {
    "matcher": {
        "match_threshold": 0.5,
    },
    "model": {
    },
    "train": {
        "epochs": 150,
        "batch": 32,
        "learning_rate": 0.0001,
        "weight_decay": 1e-4,
        "loss": {
            'use_score': True,
            'lambda_s': 1.0,
        },
        "iteration": 1,                 # number of successive registration iterations
        "save_interval": 10,            # save model every N epochs
        "loaders": 8,                   # concurrent dataset threads
        "early_stopping": 100,
    },
    "val": {
        "batch": 32,
        "sets": ["test"],
        "save": False,
        "iteration": 2,
        "loaders": 8,                   # concurrent dataset threads
    },
    'dataset': {
        'num_points': 1024,
        'sets': {
            'train': {},
        },
        # "translation_good": 0.01,  # weight at which performance measure considers translation error to be 'good'
    },
}


################################################################################
def performance(result: dict, score_loss: bool, legacy: bool = False,
                rotation_good: float = 1.5, translation_good: float = 0.01) -> float:
    """ single performance score for matcher """
    weight_at_good = np.log(0.8)
    t_alpha = weight_at_good/translation_good
    r_alpha = weight_at_good/rotation_good

    if score_loss:
        # product of precision, recall, 1 - rotation_error / sensitivity, 1 - translation_error / sensitivity
        score = np.exp(r_alpha * result['rotation_error'][-1]) * \
                np.exp(t_alpha * result['translation_error'][-1])
        if legacy:
            score *= result['precision_match'][-1] * result['recall_match'][-1]

    else:
        score = np.exp(r_alpha * result['rotation_error_nn'][-1]) * \
                np.exp(t_alpha * result['translation_error_nn'][-1])

    return score


################################################################################
def train(
        opts: argparse.Namespace,
        config: dict,
        logger: Logger = None,
):
    """
    train a point set matching network

    :param opts:    script options
    :param config:  configuration dictionary
    :param logger:  logging instance
    :return:
    """
    if logger is None:
        logger = Logger(__name__ + ':train()')
        basicConfig(level='INFO')

    tb_writer = SummaryWriter(opts.output)

    # make sure weights subdirectory exists
    (opts.output / 'weights').mkdir(parents=True, exist_ok=True)

    # get model
    ckpt = {}
    if opts.resume:
        # resume training

        # replace config dict with copy from training folder
        run_config_path = opts.output / 'config.json'
        logger.info(f'reloading config from {run_config_path}')
        with open(run_config_path, 'r') as f:
            config = json.load(f)

        model_path = opts.output / 'weights' / 'last.pt'
        logger.info(f'loading state from {model_path}')
        ckpt = torch.load(model_path)
        model = GAFAR(config['model'])
        model.load_state_dict(ckpt['model'].state_dict())

        # random generator seeds
        if not opts.renew_random_state:
            logger.info('initializing random state from checkpoint')
            rng_save_restore(ckpt['rng'])

        epoch = ckpt['epoch'] + 1
        best_epoch = ckpt['stats']['epoch']
        best_loss = ckpt['stats']['loss']
        best_precision = ckpt['stats']['precision']
        best_recall = ckpt['stats']['recall']
        best_rotation = ckpt['stats']['rotation']
        best_translation = ckpt['stats']['translation']
        best_performance = ckpt['stats']['performance']

    else:
        # check for normals in dataset, adjust model config accordingly
        if 'normals' in config['dataset']:
            config['model']['normals'] = config['dataset']['normals']

        # get model
        if opts.model:
            initial_path = Path(opts.model)
            logger.info(f'loading initial state and model config from \"{initial_path}\"')
            state = torch.load(initial_path)
            model = GAFAR(state['model'].config)
            model.load_state_dict(state['model'].state_dict())
            if not opts.renew_random_state and 'rng' in state:
                logger.info('setting random state from initial state')
                rng_save_restore(state['rng'])
            elif opts.manual_seed:
                logger.info(f'setting random state to {opts.seed}')
                rng_save_restore(opts.seed)
        else:
            model = GAFAR(config['model'])
            # seed random number generators
            if opts.manual_seed:
                logger.info(f'setting random state to {opts.seed}')
                rng_save_restore(opts.seed)

        # return full set of model parameters (i.e. with defaults) back to experiment config
        config['model'] = model.config
        logger.info(f'training model with point normals: {"YES" if config["model"]["normals"] else "NO"}')

        epoch = 0
        best_epoch = -1
        best_loss = np.inf
        best_precision = 0.
        best_recall = 0.
        best_rotation = 180.
        best_translation = np.inf
        best_performance = 0.

    logger.info(f'batch size: {config["train"]["batch"]}')

    # gpu choice/torch device
    device = get_device(opts.device, opts.gpu)
    logger.info(f'training on device: {str(device)}')

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['train']['learning_rate'],
        weight_decay=config['train']['weight_decay'],
    )

    # save initial state if not resuming
    if not opts.resume:
        ckpt = {
            'epoch': epoch,
            'model': deepcopy(model),
            'optimizer': optimizer.state_dict(),
            'stats': {
                'epoch': best_epoch,
                'loss': best_loss,
                'precision': best_precision,
                'recall': best_recall,
                'rotation': best_rotation,
                'translation': best_translation,
                'performance': best_performance,
            },
            'rng': rng_save_restore(),
        }
        initial_path = opts.output / 'weights' / f'initial_{f"{time():.8f}"[-8:]}.pt'
        logger.info(f'saving initial state to {initial_path}')
        torch.save(ckpt, initial_path)

    # loading dataset
    num_loaders = config['train']['loaders']
    training_set = ModelNet40(config['dataset'], 'train')
    training_loader = DataLoader(
        training_set,
        num_workers=num_loaders,
        prefetch_factor=max(config['train']['batch'] // num_loaders, 2),
        batch_size=config['train']['batch'],
        shuffle=True,
        drop_last=True,
        worker_init_fn=worker_random_init,
    )
    # check if validation set is a portion of the training set
    training_items = getattr(training_set, 'items', None) if 'shard' in config['dataset']['sets']['train'] else None

    validation_loader = DataLoader(
        ModelNet40(
            config['dataset'],
            'val',
            samples=training_items,
            samples_complement=True,
        ),
        num_workers=num_loaders,
        prefetch_factor=max(config['train']['batch'] // num_loaders, 2),
        batch_size=config['train']['batch'],
        shuffle=False,
        drop_last=False,
        worker_init_fn=worker_random_init,
    )

    # return full dataset config to experiment config
    config['dataset'] = training_set.config

    # loss function
    criterion = Loss(**config['train']['loss'])
    logger.info('Loss: ' + str(criterion))

    model.to(device)
    criterion.to(device)

    if not opts.resume:
        losses = ['%s/loss/loss']
        if getattr(criterion, "is_score_loss", False):
            losses.append('%s/loss/score')
        if getattr(criterion, "is_transformation_loss", False):
            losses.append('%s/loss/rot')
            losses.append('%s/loss/trans')
        if len(losses) > 2:
            tb_writer.add_custom_scalars_multilinechart(
                [x % 'train' for x in losses], title='losses')
            tb_writer.add_custom_scalars_multilinechart(
                [x % 'val' for x in losses], title='losses')

        tb_writer.add_custom_scalars_multilinechart(
            ['train/time/epoch', 'train/time/model_update', 'train/time/batch_sum'])

        # write current config to tensorboard log directory
        write_json(opts.output / 'config.json', config)

    # load remaining states
    if opts.resume:
        optimizer.load_state_dict(ckpt['optimizer'])

    if config['train']['iteration'] > 1:
        logger.info(f'model training with <{config["train"]["iteration"]}> alignment iterations')

    #
    # TRAINING
    #
    logger.info('starting model training...')
    iteration = max(1, config['train']['iteration'])
    for epoch in range(epoch, config['train']['epochs']):
        train_start_time = time()

        result = {
            'loss': np.zeros((iteration,)),
            'score_loss': np.zeros((iteration,)),
            'rotation_loss': np.zeros((iteration,)),
            'translation_loss': np.zeros((iteration,)),
            'precision': np.zeros((iteration,)),
            'recall': np.zeros((iteration,)),
            'precision_match': np.zeros((iteration,)),
            'recall_match': np.zeros((iteration,)),
            'correspondences': np.zeros((iteration,)),
            'matches_predicted': np.zeros((iteration,)),
            'matches_predicted_threshold': np.zeros((iteration,)),
            'rotation_error': np.zeros((iteration,)),
            'translation_error': np.zeros((iteration,)),
            'rotation_error_valid': np.zeros((iteration,)),
            'translation_error_valid': np.zeros((iteration,)),
            'rotation_error_nn': np.zeros((iteration,)),
            'translation_error_nn': np.zeros((iteration,)),
            'chamfer_distance': np.zeros((iteration,)),
            'chamfer_distance_valid': np.zeros((iteration,)),
            'chamfer_distance_nn': np.zeros((iteration,)),
            'wrong_score_max': np.zeros((iteration,)),
            'wrong_score_min': np.ones((iteration,)),
            'wrong_score_avg': np.zeros((iteration,)),
            'dust_bin_score': np.zeros((iteration,)),
            'time': np.zeros((iteration,)),
            'time_batch': np.zeros((iteration,)),
            'time_train': 0.0,
            'count': np.zeros((iteration,)),
            'valid': np.zeros((iteration,)),
        }
        count_wrong = np.zeros((iteration,))

        model.train()
        for source_points, target_points, correspondences, gt_rotation, gt_translation, _, _, gt_clean in \
                training_loader:
            batch_size = source_points.shape[0]
            source_points_c = source_points.to(device)
            target_points_c = target_points.to(device)
            gt_rotation_c = gt_rotation.double().to(device)
            gt_translation_c = gt_translation.double().to(device)
            rotation_estimate = torch.eye(3, dtype=torch.double, device=device).unsqueeze(0).expand(batch_size, -1, -1)
            translation_estimate = torch.zeros_like(gt_translation_c)

            ppm_valid = torch.ones((batch_size,), dtype=torch.bool, device=device)

            for idx in range(iteration):
                start_time = time()
                optimizer.zero_grad()

                scores = model(source_points_c, target_points_c)
                scores = scores['scores']
                losses = criterion(
                    scores,
                    source_points_c[:, :, :3],
                    target_points_c[:, :, :3],
                    correspondence=correspondences.long().to(device),
                    rotation=gt_rotation_c,
                    translation=gt_translation_c,
                )
                loss = losses['loss']

                result['loss'][idx] += loss.item()
                result['score_loss'][idx] += losses['s_loss'].item()
                result['rotation_loss'][idx] += losses['r_loss'].sum().item()
                result['translation_loss'][idx] += losses['t_loss'].sum().item()

                loss.backward()
                optimizer.step()

                result['time'][idx] += time() - start_time

                # calculate errors of transformation for network estimate
                rotation_error_nn_batch, translation_error_nn_batch = transformation_error(
                    losses['rotation'],
                    losses['translation'],
                    gt_rotation_c,
                    gt_translation_c,
                )

                result['chamfer_distance_nn'][idx] += chamfer_distance_modified(
                    (source_points_c[:, :, :3].double().matmul(losses['rotation'].transpose(1, 2)) +
                     losses['translation'].unsqueeze(1)),
                    target_points_c[:, :, :3].double(),
                    gt_clean[:, :, :3].double().to(device),
                    gt_clean[:, :, :3].double().to(device),
                ).sum().item()

                if criterion.has_score_loss:
                    prediction = scores_to_prediction(scores.detach(), config['matcher']['match_threshold'])

                    # non-training relevant performance metrics
                    rotation_estimate, translation_estimate = estimate_transform(
                        source_points_c[:, :, :3].double(),
                        target_points_c[:, :, :3].double(),
                        # predictions,
                        {
                            'matches1': prediction['matches1'].to(device),
                            'matching_scores1': prediction['matching_scores1'].double().to(device),
                        },
                    )

                    rotation_error_batch, translation_error_batch = transformation_error(
                        rotation_estimate.cpu(),
                        translation_estimate.cpu(),
                        gt_rotation_c.double(),
                        gt_translation_c.double(),
                    )

                    # transformations based on PPMs only valid for >= 3 point matches
                    ppm_valid = (prediction['matches1'] >= 0).sum(1) > 2
                    result['valid'][idx] += ppm_valid.sum().item()

                    overall, matched, wrong_scores = evaluate_performance(prediction, correspondences)
                    result['wrong_score_max'][idx] = max(wrong_scores['max'], result['wrong_score_max'][idx])
                    result['wrong_score_min'][idx] = min(wrong_scores['min'], result['wrong_score_min'][idx])
                    result['wrong_score_avg'][idx] += wrong_scores['avg']
                    result['precision'][idx] += overall['precision'].sum().item()
                    result['recall'][idx] += overall['recall'].sum().item()
                    result['precision_match'][idx] += matched['precision'].sum().item()
                    result['recall_match'][idx] += matched['recall'].sum().item()
                    result['rotation_error'][idx] += rotation_error_batch.sum().item()
                    result['translation_error'][idx] += translation_error_batch.sum().item()
                    result['rotation_error_valid'][idx] += rotation_error_batch[ppm_valid].sum().item()
                    result['translation_error_valid'][idx] += translation_error_batch[ppm_valid].sum().item()
                    count_wrong[idx] += wrong_scores['num_wrong']
                    result['matches_predicted'][idx] += (prediction['matches0_all'] >= 0).sum().item()
                    result['matches_predicted_threshold'][idx] += (prediction['matches0'] >= 0).sum().item()

                if not criterion.is_score_loss:
                    rotation_estimate = losses['rotation']
                    translation_estimate = losses['translation']

                chamfer_batch = chamfer_distance_modified(
                    source_points_c[:, :, :3].double().matmul(rotation_estimate.transpose(1, 2)) +
                    translation_estimate.unsqueeze(1),
                    target_points_c[:, :, :3].double(),
                    gt_clean[:, :, :3].double().to(device),
                    gt_clean[:, :, :3].double().to(device),
                    )
                result['chamfer_distance'][idx] += chamfer_batch.cpu().sum().item()

                if criterion.has_score_loss:
                    result['chamfer_distance_valid'][idx] += chamfer_batch[ppm_valid].sum().item()

                result['rotation_error_nn'][idx] += rotation_error_nn_batch.sum().item()
                result['translation_error_nn'][idx] += translation_error_nn_batch.sum().item()
                if hasattr(model, 'bin_score'):
                    result['dust_bin_score'][idx] += model.bin_score.item() * batch_size
                result['correspondences'][idx] += np.sum(correspondences.cpu().detach().numpy() >= 0).astype(np.double)
                result['count'][idx] += batch_size

                # update source point cloud, gt_rotation, gt_translation and resulting transformation
                source_points_c[:, :, :3] = \
                    (torch.matmul(source_points_c[:, :, :3].double(), rotation_estimate.transpose(1, 2)) +
                     translation_estimate.unsqueeze(1)).float()
                if source_points_c.shape[2] == 6:
                    # normals
                    source_points_c[:, :, 3:] = \
                        torch.matmul(source_points_c[:, :, 3:].double(), rotation_estimate.transpose(1, 2))
                gt_rotation_c = torch.matmul(rotation_estimate.transpose(1, 2), gt_rotation_c)
                gt_translation_c = gt_translation_c - torch.matmul(gt_rotation_c,
                                                                   translation_estimate.unsqueeze(2)).squeeze()

                result['time_batch'][idx] += time() - start_time

        result['dust_bin_score'] /= result['count']
        result['correspondences'] /= result['count']
        result['recall'] /= result['count']
        result['precision'] /= result['count']
        result['rotation_error'] = result['rotation_error'] / result['count'] / np.pi * 180.
        result['translation_error'] /= result['count']
        result['rotation_error_nn'] = result['rotation_error_nn'] / result['count'] / np.pi * 180.
        result['translation_error_nn'] /= result['count']
        if criterion.has_score_loss:
            result['valid'][result['valid'] == 0.0] = 1.0
            result['rotation_error_valid'] = result['rotation_error_valid'] / result['valid'] / np.pi * 180.
            result['translation_error_valid'] /= result['valid']
            result['chamfer_distance_valid'] /= result['valid']
            count_wrong[count_wrong == 0.0] = 1.0
            result['wrong_score_avg'] /= count_wrong
        result['precision_match'] /= result['count']
        result['recall_match'] /= result['count']
        result['matches_predicted'] /= result['count']
        result['matches_predicted_threshold'] /= result['count']
        result['chamfer_distance'] /= result['count']
        result['chamfer_distance_nn'] /= result['count']

        result['time_train'] = time() - train_start_time

        # write results to tensorboard
        log_tensorboard(tb_writer, criterion, result, epoch=epoch, mode='train')

        logger.info(f'Train {epoch:03d}:')
        log_console(result, logger, criterion=criterion, prepend='\t', mode='train')

        model.eval()
        result = val(
            config, model,
            opts=opts,
            device=device,
            criterion=criterion,
            data_loader=validation_loader,
            logger=logger,
        )

        # write results to tensorboard
        log_tensorboard(tb_writer, criterion, result, epoch=epoch, mode='val')

        # check model performance
        score_loss = getattr(criterion, 'is_score_loss', False)
        epoch_performance = performance(result, score_loss,
                                        translation_good=config['dataset']['translation_good'])
        if epoch_performance > best_performance:
            best_epoch = epoch
            best_loss = result['loss'][-1]
            best_recall = result['recall_match'][-1] if getattr(criterion, 'has_score_loss', False) else np.nan
            best_precision = result['precision_match'][-1] if getattr(criterion, 'has_score_loss', False) else np.nan
            best_rotation = result['rotation_error'][-1] if score_loss else result['rotation_error_nn'][-1]
            best_translation = result['translation_error'][-1] if score_loss else result['translation_error_nn'][-1]
            best_performance = epoch_performance

        ckpt = {
            'epoch': epoch,
            'model': deepcopy(model),
            'optimizer': optimizer.state_dict(),
            'stats': {
                'epoch': best_epoch,
                'loss': best_loss,
                'precision': best_precision,
                'recall': best_recall,
                'rotation': best_rotation,
                'translation': best_translation,
                'performance': best_performance,
            },
            'rng': rng_save_restore(),
        }

        torch.save(ckpt, opts.output / 'weights' / 'last.pt')

        if config['train']['save_interval'] > 0 and (epoch % config['train']['save_interval'] == 0):
            # save current model
            model_path = opts.output / 'weights' / f'epoch_{epoch:d}.pt'
            torch.save(ckpt, model_path)

        if epoch == best_epoch:
            # save because best
            torch.save(ckpt, opts.output / 'weights' / 'best.pt')

        logger.info(f'Val   {epoch:03d}:  ({epoch_performance:.5f})')
        log_console(result, logger, criterion=criterion, prepend='\t')

        if utils.sigint.sigint_status:
            # SIGINT caught, end training (but do final evaluation and logging_
            logger.warning('SIGINT caught, ending training!')
            break

        if 0 < config['train']['early_stopping'] < epoch - best_epoch:
            # no improvement, stop early
            logger.warning(f'no improvement over the last {epoch - best_epoch} epochs, stop training')
            break

    # copy best model to main experiment directory
    if best_epoch > 0:
        if getattr(criterion, "is_score_loss", False):
            logger.info(f'best model with performance {best_performance:.3f}, '
                        f'and validation metrics: recall {best_recall:.3f}, precision {best_precision:.3f} ' +
                        f'residual errors rotation {best_rotation:.3f}, translation {best_translation:.3f} '
                        f'in epoch {best_epoch:03d} (loss {best_loss:.6f})')
        else:
            logger.info(f'best model with performance {best_performance:.3f},'
                        f'validation residual rotation error {best_rotation:.3f} and translation ' +
                        f'error {best_translation:.3f} in epoch {best_epoch:03d} (loss {best_loss:.6f})')

        copyfile(opts.output / 'weights' / 'best.pt', (opts.output / getattr(opts, "name", "model")).with_suffix(".pt"))
        write_json(
            (opts.output / getattr(opts, "name", "model")).with_suffix(".json"),
            {'matcher': config['matcher'], 'model': config['model']})

        # load best model
        ckpt = torch.load((opts.output / getattr(opts, "name", "model")).with_suffix(".pt"))
        model.load_state_dict(ckpt['model'].state_dict())
        result = val(
            config,
            model,
            opts=opts,
            device=device,
            criterion=criterion,
            data_loader=validation_loader,
        )

        epoch_performance = performance(result, getattr(criterion, 'is_score_loss', False),
                                        translation_good=config['dataset']['translation_good'])
        logger.info(f'Val   {best_epoch:03d}:  ({epoch_performance:.5f})')
        log_console(result, logger, criterion=criterion)

    else:
        logger.warning('FAILED to train model')


@torch.no_grad()
def val(
        config: dict,
        model: Union[torch.nn.Module, Path],
        opts: argparse.Namespace = None,
        device: torch.device = None,
        criterion: torch.nn.Module = None,
        data_loader: torch.utils.data.DataLoader = None,
        logger: Logger = None,
) -> dict:
    """
    test an epoch/a full run on the specified partitions

    :param config: configuration dict
    :param opts: command line options
    :param model: trained model to evaluate
    :param device:
    :param criterion: loss function
    :param data_loader:
    :param logger:
    :return:
    """
    if not logger:
        logger = Logger(__name__ + ':val()')
        basicConfig(level='INFO')

    is_training = True
    if isinstance(model, Path):
        # standalone call
        is_training = False

        # load model
        logger.info(f'loading model state from {model}')
        ckpt = torch.load(model)
        if hasattr(ckpt['model'], 'config'):
            model = GAFAR(ckpt['model'].config)
            model.load_state_dict(ckpt['model'].state_dict())
        else:
            model = GAFAR(ckpt['config'])
            model.load_state_dict(ckpt['model'])

        config['model'] = model.config

        logger.info(f'feature dimension <{config["model"]["feature_dimension"]}>, '
                    f'matching attention <{config["model"]["matcher"]["attention"]}>'
                    )

        device = get_device(opts.device, opts.gpu)
        model.to(device)

        logger.info(f'batch size: {config["val"]["batch"]}')
        logger.info(f'match threshold: {config["matcher"]["match_threshold"]}')

        # seed torch random number generator
        if opts and opts.manual_seed:
            if opts.seed < 0 and 'rng' in ckpt:
                logger.info(f'setting random seeds from checkpoint')
                rng_save_restore(ckpt['rng'])
            else:
                logger.info(f'setting random seeds to {opts.seed}')
                rng_save_restore(opts.seed)

        # get default loss
        if not criterion:
            criterion = Loss()
            criterion.to(device)

    data = [data_loader] if data_loader else config['val']['sets']
    result = None   # so the linter does not complain
    for partition in data:
        # load data set/partition
        if not isinstance(partition, DataLoader):
            logger.info(f'loading datasets for <{partition.upper()}>')
            if 'normals' in config['model']:
                logger.info(f'testing model with point normals: {"YES" if config["model"]["normals"] else "NO"}')
                config['dataset']['normals'] = config['model']['normals']

            data_loader = DataLoader(
                ModelNet40(config['dataset'], partition),
                num_workers=config['val']['loaders'],
                prefetch_factor=max(config['val']['batch'] // config['val']['loaders'], 2),
                batch_size=config['val']['batch'],
                shuffle=False,
                drop_last=False,
                worker_init_fn=worker_random_init,
            )

        iteration = max(1, config['val']['iteration'])
        result = {
            'loss': np.zeros((iteration,)),
            'score_loss': np.zeros((iteration,)),
            'rotation_loss': np.zeros((iteration,)),
            'translation_loss': np.zeros((iteration,)),
            'precision': np.zeros((iteration,)),
            'recall': np.zeros((iteration,)),
            'precision_match': np.zeros((iteration,)),
            'recall_match': np.zeros((iteration,)),
            'correspondences': np.zeros((iteration,)),
            'matches_predicted': np.zeros((iteration,)),
            'matches_predicted_threshold': np.zeros((iteration,)),
            'rotation_error': np.zeros((iteration,)),
            'translation_error': np.zeros((iteration,)),
            'rotation_error_valid': np.zeros((iteration,)),
            'translation_error_valid': np.zeros((iteration,)),
            'rotation_error_nn': np.zeros((iteration,)),
            'translation_error_nn': np.zeros((iteration,)),
            'chamfer_distance': np.zeros((iteration,)),
            'chamfer_distance_nn': np.zeros((iteration,)),
            'chamfer_distance_valid': np.zeros((iteration,)),
            'wrong_score_max': np.zeros((iteration,)),
            'wrong_score_min': np.ones((iteration,)),
            'wrong_score_avg': np.zeros((iteration,)),
            'dust_bin_score': np.zeros((iteration,)),
            'count': np.zeros((iteration,)),
            'valid': np.zeros((iteration,)),
        }
        count_wrong = np.zeros((iteration,))

        # evaluation loop
        model.eval()
        for source_points, target_points, correspondences, gt_rotation, gt_translation, _, _, gt_clean in data_loader:
            batch_size = source_points.shape[0]
            source_points_c = source_points.to(device=device)
            target_points_c = target_points.to(device=device)
            rotation_estimate_res = torch.eye(3, dtype=torch.double, device=device).unsqueeze(0).expand(batch_size, -1,
                                                                                                        -1)
            translation_estimate_res = torch.zeros((batch_size, 3), dtype=torch.double, device=device)

            gt_rotation_c = gt_rotation.double().to(device)
            gt_translation_c = gt_translation.double().to(device)

            rotation_estimate = torch.eye(3, dtype=torch.double, device=device).unsqueeze(0).expand(batch_size, -1, -1)
            translation_estimate = torch.zeros_like(gt_translation_c)

            ppm_valid = torch.ones((batch_size,), dtype=torch.bool, device=device)

            # registration iteration
            for idx in range(iteration):
                scores = model(source_points_c, target_points_c)
                scores = scores['scores']
                losses = criterion(
                    scores,
                    source_points_c[:, :, :3],
                    target_points_c[:, :, :3],
                    correspondence=correspondences.long().to(device),
                    rotation=gt_rotation_c,
                    translation=gt_translation_c,
                )
                loss = losses['loss']

                result['loss'][idx] += loss.item()
                result['score_loss'][idx] += losses['s_loss'].item()
                result['rotation_loss'][idx] += losses['r_loss'].sum().item()
                result['translation_loss'][idx] += losses['t_loss'].sum().item()

                rotation_error_nn_batch, translation_error_nn_batch = transformation_error(
                    losses['rotation'],
                    losses['translation'],
                    gt_rotation_c,
                    gt_translation_c,
                )

                chamfer_batch = chamfer_distance_modified(
                    (source_points_c[:, :, :3].double().matmul(losses['rotation'].transpose(1, 2)) +
                     losses['translation'].unsqueeze(1)),
                    target_points_c[:, :, :3].double(),
                    gt_clean[:, :, :3].double().to(device),
                    gt_clean[:, :, :3].double().to(device),
                ).sum().item()
                result['chamfer_distance_nn'][idx] += chamfer_batch

                if criterion.has_score_loss:
                    prediction = scores_to_prediction(scores.detach(), config['matcher']['match_threshold'])

                    rotation_estimate, translation_estimate = estimate_transform(
                        source_points_c[:, :, :3],
                        target_points_c[:, :, :3],
                        # predictions,
                        {
                            'matches1': prediction['matches1'].to(device),
                            'matching_scores1': prediction['matching_scores1'].to(device),
                        },
                    )

                    # transformations based on PPMs only valid for >= 3 point matches
                    ppm_valid = (prediction['matches1'] >= 0).sum(1) > 2
                    n_valid_batch = ppm_valid.sum().item()

                    rotation_error_batch, translation_error_batch = transformation_error(
                        rotation_estimate.double(),
                        translation_estimate.double(),
                        gt_rotation_c,
                        gt_translation_c,
                    )

                    overall, matched, wrong_scores = evaluate_performance(prediction, correspondences)
                    result['wrong_score_max'][idx] = max(wrong_scores['max'], result['wrong_score_max'][idx])
                    result['wrong_score_min'][idx] = min(wrong_scores['min'], result['wrong_score_min'][idx])
                    result['wrong_score_avg'][idx] += wrong_scores['avg']
                    result['precision'][idx] += overall['precision'].sum().item()
                    result['recall'][idx] += overall['recall'].sum().item()
                    result['rotation_error'][idx] += rotation_error_batch.sum().item()
                    result['translation_error'][idx] += translation_error_batch.sum().item()
                    result['rotation_error_valid'][idx] += rotation_error_batch[ppm_valid].sum().item()
                    result['translation_error_valid'][idx] += translation_error_batch[ppm_valid].sum().item()
                    result['precision_match'][idx] += matched['precision'].sum().item()
                    result['recall_match'][idx] += matched['recall'].sum().item()
                    result['matches_predicted'][idx] += (prediction['matches0_all'] >= 0).sum().item()
                    result['matches_predicted_threshold'][idx] += (prediction['matches0'] >= 0).sum().item()
                    result['valid'][idx] += n_valid_batch
                    count_wrong[idx] += wrong_scores['num_wrong']

                if not criterion.is_score_loss:
                    rotation_estimate = losses['rotation']
                    translation_estimate = losses['translation']

                chamfer_batch = chamfer_distance_modified(
                    (source_points_c[:, :, :3].double().matmul(rotation_estimate.transpose(1, 2)) +
                     translation_estimate.unsqueeze(1)),
                    target_points_c[:, :, :3].double(),
                    gt_clean[:, :, :3].double().to(device),
                    gt_clean[:, :, :3].double().to(device),
                )
                result['chamfer_distance'][idx] += chamfer_batch.sum().item()

                if criterion.has_score_loss:
                    result['chamfer_distance_valid'][idx] += chamfer_batch[ppm_valid].sum().item()

                # non batch normalized average accuracy
                result['correspondences'][idx] += (correspondences.cpu().detach().numpy() >= 0).sum().item()
                result['rotation_error_nn'][idx] += rotation_error_nn_batch.sum().item()
                result['translation_error_nn'][idx] += translation_error_nn_batch.sum().item()
                result['count'][idx] += batch_size

                # update source point cloud, gt_rotation, gt_translation and resulting transformation
                source_points_c[:, :, :3] = \
                    (torch.matmul(source_points_c[:, :, :3].double(), rotation_estimate.transpose(1, 2)) +
                     translation_estimate.unsqueeze(1)).float()
                if source_points_c.shape[2] == 6:
                    # normals
                    source_points_c[:, :, 3:] = \
                        torch.matmul(source_points_c[:, :, 3:].double(), rotation_estimate.transpose(1, 2))
                gt_rotation_c = torch.matmul(rotation_estimate.transpose(1, 2), gt_rotation_c)
                gt_translation_c = gt_translation_c - torch.matmul(gt_rotation_c,
                                                                   translation_estimate.unsqueeze(2)).squeeze()
                rotation_estimate_res = torch.matmul(rotation_estimate_res, rotation_estimate)
                translation_estimate_res = \
                    torch.matmul(rotation_estimate,
                                 translation_estimate_res.unsqueeze(2)).squeeze() + translation_estimate

        result['correspondences'] /= result['count']
        result['precision'] /= result['count']
        result['recall'] /= result['count']
        result['rotation_error'] = result['rotation_error'] / result['count'] / np.pi * 180.
        result['translation_error'] /= result['count']
        result['rotation_error_nn'] = result['rotation_error_nn'] / result['count'] / np.pi * 180.
        result['translation_error_nn'] /= result['count']
        if criterion.has_score_loss:
            result['valid'][result['valid'] == 0.0] = 1.0
            result['rotation_error_valid'] = result['rotation_error_valid'] / result['valid'] / np.pi * 180.
            result['translation_error_valid'] /= result['valid']
            result['chamfer_distance_valid'] /= result['valid']
            count_wrong[count_wrong == 0.] = 1.0
            result['wrong_score_avg'] /= count_wrong

        result['precision_match'] /= result['count']
        result['recall_match'] /= result['count']
        result['matches_predicted'] /= result['count']
        result['matches_predicted_threshold'] /= result['count']
        result['chamfer_distance'] /= result['count']
        result['chamfer_distance_nn'] /= result['count']

        if not is_training:
            # write output
            logger.info(f'Testing  partition \"{partition.upper()}\":')
            log_console(result, logger, criterion)

    return result


def get_args():
    parser = argparse.ArgumentParser('train/test point cloud matching on small (ModelNet40 2048 point) point clouds')
    parser.add_argument('config', metavar='config.json', help='config file')
    parser.add_argument('output', metavar='output/path', help='output path for trained models, logs, etc.')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size from command line.')
    parser.add_argument('-d', '--dataset', default=None, type=str,
                        help='dataset config file. Supersedes dataset in main config file.')
    parser.add_argument('--deterministic', action='store_true', help='Set CUDA to deterministic execution.')
    parser.add_argument('--device', type=str, choices=['cuda', 'gpu', 'cpu'], default='cuda',
                        help='train/evaluate on GPU or CPU.')
    parser.add_argument('--gpu', type=int, default=None, help='PyTorch GPU id to use if multiple GPUs in system.')
    parser.add_argument('--iteration', type=int, default=None, help='Override number of training/testing iterations.')
    parser.add_argument('-m', '--model', default=None, type=str,
                        help='Initialize model from pretrained weights or Resume this checkpoint/experiment.')
    parser.add_argument('-r', '--resume', action='store_true', help='Resume last training instead of starting new one.')
    parser.add_argument('-t', '--threshold', type=float, default=None, help='Override score threshold of matches.')
    parser.add_argument('-v', '--validate', action='store_true',
                        help='Validate model (instead of doing a training run).')
    # random seed
    parser.add_argument('--renew-random-state', action='store_true',
                        help='Use new random state when resuming or initializing from a checkpoint.')
    parser.add_argument('--manual-seed', action='store_true', help='Use manual seeding of random generators.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Manual random seed to use (if --manual_seed is set).'
                             'In standalone validation, use rng state of checkpoint if seed == -1.')

    return parser.parse_args()


def main(opts: argparse.Namespace):
    """ run train/validation script """
    opts.config = Path(opts.config)
    if opts.model:
        opts.model = Path(opts.model)
    if opts.dataset:
        opts.dataset = Path(opts.dataset)
    opts.output = get_tensorboard_log_path(Path(opts.output), resume=opts.resume)
    if not opts.output.exists():
        # catch if --resume flag is set in empty/new directory
        opts.output.mkdir(parents=True)

    with open(opts.config, 'r') as f:
        config = merge_dict(_default_config, json.load(f))

    # get dataset
    if opts.dataset:
        with open(opts.dataset, 'r') as f:
            if not opts.validate:   # training
                config = merge_dict(json.load(f), config)
            else:
                config = merge_dict(config, json.load(f))

    logger = get_logger(opts.output / 'run.log', 'INFO')
    logger.info(f'read config from {opts.config}')
    if opts.dataset:
        logger.info(f'using dataset {opts.dataset}')
    logger.info(f'resuming from {opts.output}' if opts.resume else f'saving new run to {opts.output}')

    # deterministic behaviour of cuda
    if opts.deterministic or ('deterministic' in config and config['deterministic']):
        logger.info('CUDA deterministic behaviour')
        # torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        config['deterministic'] = True

    if opts.batch_size is not None:
        logger.info(f'Overriding batch size: {opts.batch_size}')
        config['train']['batch'] = opts.batch_size
        config['val']['batch'] = opts.batch_size

    if opts.iteration is not None:
        logger.info(f'Overriding number of training iterations: {opts.iteration}')
        config['train']['iteration'] = opts.iteration
        config['val']['iteration'] = opts.iteration

    if opts.threshold is not None:
        logger.info(f'Overriding score threshold for matches: {opts.threshold}')
        config['matcher']['match_threshold'] = opts.threshold

    if not opts.validate:
        # register handler for sigint to make training gracefully interruptable
        signal.signal(signal.SIGINT, utils.sigint.sigint_handler)
        opts.name = opts.config.name

        try:
            train(
                opts,
                config,
                logger=logger,
            )
        except Exception as e:
            # save especially CUDA errors to log file
            logger.exception(e)

    else:
        # write config to output directory
        write_json(opts.output / 'config.json', config)

        val(
            config,
            opts=opts,
            model=opts.model,
            logger=logger,
        )


################################################################################
if __name__ == '__main__':
    # command line arguments
    args = get_args()

    # run script
    main(args)
