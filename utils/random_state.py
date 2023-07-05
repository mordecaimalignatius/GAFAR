"""
utilities for torch etc. random generators

Ludwig Mohr
ludwig.mohr@icg.tugraz.at

"""

import numpy as np
import torch
import random
from typing import Union

from logging import getLogger


################################################################################
def worker_random_init(worker_id):
    """
    init function for numpy random generators in data loader workers

    :param worker_id:
    :return:
    """
    worker_info = torch.utils.data.get_worker_info()

    if worker_info is not None:
        np.random.seed(worker_info.seed % (2**32 - 1))

    else:
        np.random.seed((torch.initial_seed() % (2**32 - 1) + worker_id) % (2**32 - 1))


################################################################################
def get_random_seed() -> int:
    return np.random.randint(0, np.iinfo(np.int32).max)


################################################################################
def get_random_state() -> dict:
    return {
        'torch': torch.random.get_rng_state(),
        'cuda': torch.cuda.random.get_rng_state(),
        'cuda_all': torch.cuda.random.get_rng_state_all(),
        'numpy': np.random.get_state(),
        'python': random.getstate(),
    }


def rng_save_restore(state: Union[dict, torch.Tensor, int] = None) -> dict:
    if state is None:
        state = get_random_state()

    else:
        logger = getLogger(__name__)
        if isinstance(state, dict):
            if 'torch' in state:
                logger.info('setting torch random state')
                torch.random.set_rng_state(state['torch'])
            if 'cuda' in state:
                logger.info('setting torch.cuda random state')
                torch.cuda.random.set_rng_state(state['cuda'])
            if 'cuda_all' in state:
                try:
                    logger.info('setting all torch.cuda random states')
                    torch.cuda.random.set_rng_state_all(state['cuda_all'])
                except IndexError:
                    logger.warning('Failed to reload all random states [IndexError]. This is sloppy for migrating '
                                   'jobs between machines with different number of CUDA devices.')
            if 'numpy' in state:
                logger.info('setting numpy random convenience random state')
                np.random.set_state(state['numpy'])
            if 'python' in state:
                logger.info('setting python random state')
                random.setstate(state['python'])

        elif isinstance(state, torch.Tensor):
            try:
                logger.info('setting via torch.set_rng_state()')
                torch.set_rng_state(state)
            except RuntimeError:
                logger.exception('failed to set torch random generator state')

        elif isinstance(state, int):
            logger.info(f'seeding random states with seed {state}')
            torch.manual_seed(state % (2**32 - 1))
            np.random.seed(state % (2**32 - 1))
            random.seed(state % (2**32 - 1))
            state = get_random_state()

        else:
            logger.warning(f'can\'t handle state of type {str(type(state))}')

    return state
