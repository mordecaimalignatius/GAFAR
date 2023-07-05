"""
miscellaneous general helper functions

Ludwig Mohr
ludwig.mohr@icg.tugraz.at

"""
import torch
from copy import copy


################################################################################
def merge_dict(dict_1, dict_2) -> dict:
    """
    recursively merge two dictionaries. returns a new dictionary

    :param dict_1:
    :param dict_2: takes precedence for duplicate entries/keys
    :return:
    """
    merged = copy(dict_2)

    for key in dict_1.keys():
        if key in merged:
            if isinstance(merged[key], dict) and isinstance(dict_1[key], dict):
                merged[key] = merge_dict(dict_1[key], merged[key])

        else:
            merged[key] = copy(dict_1[key])

    return merged


################################################################################
def merge_dict_default(default: dict, other: dict) -> dict:
    """ same as merge_dict, but the output will only contain keys present default dict """
    merged = copy(default)

    for key in other.keys():
        if key in merged:
            if isinstance(merged[key], dict) and isinstance(other[key], dict):
                merged[key] = merge_dict_default(merged[key], other[key])

            else:
                merged[key] = copy(other[key])

    return merged


################################################################################
def get_device(device: str = 'cuda', gpu: int = None) -> torch.device:
    """
    return correct torch device
    expects fields:
        "gpu": gpu id
        "device": [gpu, cuda, cpu]

    """

    if not torch.cuda.is_available() or 'cpu' in device:
        return torch.device('cpu')

    return torch.device(f'cuda:{gpu}') if gpu is not None else torch.device('cuda')
