"""
common utilities


Ludwig Mohr
ludwig.mohr@icg.tugraz.at

"""

import json

from pathlib import Path


################################################################################
def read_json(
        json_path: Path,
        ):
    """
    read configuration parameters from json file (assumed to be first argument on commandline).

    """

    if not json_path.suffix == '.json':
        json_path = json_path.with_suffix('.json')

    with open(json_path, 'r') as f:
        conf = json.load(f)

    return conf


def write_json(
    json_path: Path,
    config: dict,
):
    """
    write the configuration parameters used to create the data archive to a json file

    parameters:
        json_path       : path of output file
        config:         : dictionary of configuration parameters

    """
    with open(json_path, 'w') as f:
        json.dump(
            config,
            f,
            indent=4,
        )
