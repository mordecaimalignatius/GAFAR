"""
tools to make logging simpler/main scripts more clean and concise

Ludwig Mohr
ludwig.mohr@icg.tugraz.at

"""
import logging
import parse
from pathlib import Path
import numpy as np
import torch.utils.tensorboard
from typing import Optional, Union
from numbers import Number


################################################################################
def get_logger(
        path: Union[Path, str, None],
        level: str = 'INFO',
        colored: bool = False,
        fmt: str = '%(asctime)s [%(levelname)s] %(name)s - %(message)s',
):
    """

    :param path: output file for logger
    :param level: minimum log level
    :param colored: use colored console output
    :param fmt: format string passed to logging.Formatter
    :return: logger
    """
    if path is not None:
        if isinstance(path, str):
            path = Path(path)

        log_path = path.parent
        log_path.mkdir(parents=True, exist_ok=True)

        if not path.suffix == '.log':
            path = path.with_suffix('.log')

    log_level_numeric = getattr(logging, level)
    logger = logging.getLogger()
    logger.setLevel(log_level_numeric)
    if colored:
        try:
            import coloredlogs
            coloredlogs.install(level=log_level_numeric, logger=logger)
        except ImportError:
            pass

    handler = logging.FileHandler(path)
    formatter = logging.Formatter(fmt=fmt)
    if not logger.hasHandlers():
        # set a stream handler if it does not exist
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)

    if path is not None:
        handler = logging.FileHandler(path)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if path is not None:
        logger.info(f'Saving output and logs to {path}')
    else:
        logger.info('not saving to file')

    return logger


################################################################################
def get_tensorboard_log_path(path: Union[Path, str], resume: bool = False, val: bool = False) -> Path:
    """
    return the next/a tensorboard logging path
    $path/runs/expN

    if runs does not exist, returns $path/runs/exp01
    otherwise, checks for exp{} folders in runs and returns the next higher exp{} sub-folder

    does not create folders

    :param path:
    :param resume: if True, return last path instead of next increment
    :return: path to training directory
    """
    if isinstance(path, str):
        path = Path(path)
    path = path.absolute()
    if not val and 'runs' in path.parts:
        # get runs directory
        last_name = ''
        while path.name != 'runs':
            last_name = path.name
            path = path.parent

        # return experiment directory
        if resume:
            return path / last_name

    elif val and path.name.startswith('exp'):
        # evaluation, base directory of experiment runs/expX provided
        path = path / 'val'
    elif not val:
        path = path / 'runs'

    if not path.exists():
        return path / 'exp0'

    dirs = [x for x in path.iterdir()]
    parser = parse.compile('exp{:d}')

    nums = [x[0] for x in [parser.parse(x.name) for x in dirs] if x is not None]
    last_dir = max(nums) if len(nums) > 0 else 0

    if not (resume and len(dirs)):
        last_dir += 1

    return path / f'exp{last_dir:d}'


################################################################################
def write_scalars(
        summary_writer: torch.utils.tensorboard.SummaryWriter,
        tag: str,
        value: Union[Number, np.ndarray, list[Number, ...], tuple[Number, ...]],
        epoch: Optional[int] = None,
) -> torch.utils.tensorboard.SummaryWriter:
    """
    writes elements from results to SummaryWriter object.
    which elements from results are used and under which keys they are written is controlled by dictionary to_log.
    automatically switches between add_scalar and add_scalars, depending on the value in results being a single value
    or an iterable

    :param summary_writer:
    :param tag: tag used in summary_writer
    :param value: value/values to write. automatic arbitration between and treatment of add_scalar and add_scalars
    :param epoch: epoch for logging time series
    :return:
    """
    if isinstance(value, (np.ndarray, tuple, list)):
        if len(value) == 1:
            summary_writer.add_scalar(tag, value[0], global_step=epoch)
        else:
            summary_writer.add_scalars(tag, array2dict(value), global_step=epoch)

    else:
        summary_writer.add_scalar(tag, value, global_step=epoch)

    return summary_writer


def array2dict(array: np.ndarray) -> dict:
    return {f'i{x}': y for x, y in enumerate(array)}


def array2str(array: np.ndarray, p='.4f') -> str:
    return "/".join([f'{x:{p}}' for x in array])


################################################################################
def matching_results_to_tensorboard(
        writer: torch.utils.tensorboard.SummaryWriter,
        criterion: torch.nn.Module,
        results: dict,
        epoch: int = None,
        mode: str = 'train',    # train, val
):
    """ get excessive tensorboard logging out of main script """
    if mode != 'val':
        mode = 'train'

    write_scalars(writer, f'{mode}/loss/loss', results['loss'], epoch)
    if getattr(criterion, "has_score_loss", False):
        write_scalars(writer, f'{mode}/loss/score', results['score_loss'], epoch)
    if getattr(criterion, "has_transformation_loss", False):
        write_scalars(writer, f'{mode}/loss/rot', results['rotation_loss'], epoch)
        write_scalars(writer, f'{mode}/loss/trans', results['translation_loss'], epoch)

    write_scalars(writer, f'{mode}/metrics/chamfer', results['chamfer_distance'], epoch)
    if getattr(criterion, "has_score_loss", False):
        if mode == 'train' and 'dust_bin_score' in results:
            write_scalars(writer, f'{mode}/misc/dust_bin', results['dust_bin_score'], epoch)

        write_scalars(writer, f'{mode}/metrics/precision_all', results['precision'], epoch)
        write_scalars(writer, f'{mode}/metrics/recall_all', results['recall'], epoch)
        write_scalars(writer, f'{mode}/metrics/precision', results['precision_match'], epoch)
        write_scalars(writer, f'{mode}/metrics/recall', results['recall_match'], epoch)
        write_scalars(writer, f'{mode}/metrics/matches', results['matches_predicted'], epoch)
        write_scalars(writer, f'{mode}/metrics/matches_threshold', results['matches_predicted_threshold'], epoch)
        write_scalars(writer, f'{mode}/metrics/rotation_error', results['rotation_error'], epoch)
        write_scalars(writer, f'{mode}/metrics/translation_error', results['translation_error'], epoch)
        write_scalars(writer, f'{mode}/misc/wrong_scores_max', results['wrong_score_max'], epoch)
        write_scalars(writer, f'{mode}/misc/wrong_scores_min', results['wrong_score_min'], epoch)
        write_scalars(writer, f'{mode}/misc/wrong_scores_avg', results['wrong_score_avg'], epoch)

        write_scalars(writer, f'{mode}/metrics/rotation_error_valid', results['rotation_error_valid'], epoch)
        write_scalars(writer, f'{mode}/metrics/translation_error_valid', results['translation_error_valid'], epoch)
        write_scalars(writer, f'{mode}/metrics/chamfer_valid', results['chamfer_distance_valid'], epoch)
        write_scalars(writer, f'{mode}/misc/valid', results['valid'], epoch)

        if 'registration_recall' in results:
            write_scalars(writer, f'{mode}/metrics/registration_recall', results['registration_recall'], epoch)
            write_scalars(
                writer, f'{mode}/metrics/registration_recall_valid', results['registration_recall_valid'], epoch)

        write_scalars(writer, f'{mode}/metrics/inlier_ratio', results['inlier_ratio'], epoch)
        write_scalars(writer, f'{mode}/metrics/inlier_ratio_valid', results['inlier_ratio_valid'], epoch)

    write_scalars(writer, f'{mode}/metrics/rotation_error_nn', results['rotation_error_nn'], epoch)
    write_scalars(writer, f'{mode}/metrics/translation_error_nn', results['translation_error_nn'], epoch)
    write_scalars(writer, f'{mode}/metrics/chamfer_nn', results['chamfer_distance_nn'], epoch)
    if 'registration_recall_nn' in results:
        write_scalars(writer, f'{mode}/metrics/registration_recall_nn', results['registration_recall_nn'], epoch)
    write_scalars(writer, f'{mode}/misc/correspondences', results['correspondences'], epoch)
    write_scalars(writer, f'{mode}/misc/count', results['count'][-1], epoch)

    if mode == 'train':
        write_scalars(writer, f'{mode}/time/epoch', results['time_train'] / 60., epoch)
        write_scalars(writer, f'{mode}/time/model_update', results['time'] / 60., epoch)
        write_scalars(writer, f'{mode}/time/batch_sum', results['time_batch'] / 60., epoch)
        write_scalars(writer, f'{mode}/time/update', results['time_update'] / 60., epoch)


################################################################################
def matching_log_result(result: dict, logger: logging.Logger, criterion=None, prepend: str = '', mode: str = 'val'):
    """ log results to Logger instance """
    if mode != 'val':
        mode = 'train'

    loss_str = ''
    if criterion:
        loss_str += f'Loss {array2str(result["loss"], p="10.2f")}   '
        if getattr(criterion, 'has_score_loss', False):
            loss_str += f'score: {array2str(result["score_loss"], p="10.2f")}   '
        if getattr(criterion, 'has_transformation_loss', False):
            loss_str += f'rot: {array2str(result["rotation_loss"], p="10.2f")}   ' \
                        f'trans: {array2str(result["translation_loss"], p="10.2f")}   '

    loss_str += f'#reg: {int(result["count"][0]):d}  '

    if criterion and getattr(criterion, 'has_score_loss', False):
        loss_str += f'#val: {array2str(result["valid"].astype(int), p="d")} ' \
                    f'[{array2str(result["valid"]/result["count"][-1], p=".2%")}]'
    logger.info(prepend + loss_str)

    if getattr(criterion, 'has_score_loss', False):
        metrics = prepend + \
            f'P  : {array2str(result["precision"], p="8.4f")}  '\
            f'R  : {array2str(result["recall"], p="8.4f")}  '\
            f'PM : {array2str(result["precision_match"], p="8.4f")}  '\
            f'RM : {array2str(result["recall_match"], p="8.4f")}  '\
            f'C  : {result["correspondences"][0]:8.2f}  '\
            f'M  : {array2str(result["matches_predicted"], p="8.2f")}  '\
            f'M/t: {array2str(result["matches_predicted_threshold"], p="8.2f")}  '

        logger.info(metrics)

    metric_str = ''
    if getattr(criterion, 'has_score_loss', False):
        metric_str += f'Re : {array2str(result["rotation_error"], p="8.5f")}  ' \
                      f'Te : {array2str(result["translation_error"], p="8.5f")}  ' \
                      f'CD : {array2str(result["chamfer_distance"], p="8.6f")}  ' \
                      f'ReV: {array2str(result["rotation_error_valid"], p="8.5f")}  ' \
                      f'TeV: {array2str(result["translation_error_valid"], p="8.5f")}  ' \
                      f'CDV: {array2str(result["chamfer_distance_valid"], p="8.6f")}  '
    if getattr(criterion, 'has_transformation_loss', False):
        metric_str += f'ReN: {array2str(result["rotation_error_nn"], p="8.5f")}  ' \
                      f'TeN: {array2str(result["translation_error_nn"], p="8.5f")}  ' \
                      f'CDN: {array2str(result["chamfer_distance_nn"], p="8.6f")}'
    logger.info(prepend + metric_str)

    # registration recall and inlier ratio
    metric_str = ''
    if getattr(criterion, 'has_score_loss', False):
        if 'registration_recall' in result:
            metric_str += f'RR : {array2str(result["registration_recall"], p="8.3f")}  ' \
                          f'RRV: {array2str(result["registration_recall_valid"], p="8.3f")}  '
    if 'registration_recall_nn' in result:
        metric_str += f'RRN: {array2str(result["registration_recall_nn"], p="8.3f")}  '
    if getattr(criterion, 'has_score_loss', False):
        metric_str += f'IR : {array2str(result["inlier_ratio"], p="8.3f")}  '\
                      f'IRV: {array2str(result["inlier_ratio_valid"], p="8.3f")}  '
    if len(metric_str):
        logger.info(prepend + metric_str)

    general_info_str = ''
    if criterion.has_score_loss:
        general_info_str += f'Wmax: {array2str(result["wrong_score_max"])}  '\
                          f'Wmin: {array2str(result["wrong_score_min"])}  '\
                          f'Wavg: {array2str(result["wrong_score_avg"])}  '
        if 'dust_bin_score' in result:
            general_info_str += f'DB: {array2str(result["dust_bin_score"], p="4.2f")}  '

    if mode == 'train':
        general_info_str += f'tt: {result["time_train"] / 60.:.2f}  '\
                            f'mu: {array2str(result["time"] / 60., p=".2f")}  '\
                            f'bt: {array2str(result["time_batch"] / 60., p=".2f")}   '\
                            f'ut: {array2str(result["time_update"] / 60., p=".2f")}'

    logger.info(prepend + general_info_str)
