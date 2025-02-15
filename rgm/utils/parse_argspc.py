import argparse
from rgm.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from pathlib import Path


def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--cfg', dest='cfg_file', action='append',
                        help='an optional config file', default=None, type=str)
    parser.add_argument('--batch', dest='batch_size',
                        help='batch size', default=18, type=int)
    parser.add_argument('--epoch', dest='epoch',
                        help='epoch number', default=None, type=int)
    parser.add_argument('--model', dest='model',
                        help='model name', default=None, type=str)
    parser.add_argument('--dataset', dest='dataset',
                        help='dataset name', default=None, type=str)    # warning: use changed
    parser.add_argument('--loader', dest='data_loader',
                        help='number of dataloaders', default=None, type=int)
    parser.add_argument('--output', type=str, default=None, help='output path override')
    parser.add_argument('--reg_iter', type=int, default=1, help='registration iteration override')
    parser.add_argument('--seed', type=int, default=None, help='random seed to use.')
    parser.add_argument('--non_deterministic', action='store_true', help='deactivate deterministic testing')
    parser.add_argument('--threshold', type=float, default=0.1, help='Score matrix threshold for matches.')
    parser.add_argument('--matches-min', type=int, default=30,
                        help='minimum number of matches to consider a registration successful.')
    parser.add_argument('--discard-bin', action='store_true',
                        help='Discard dust bin row/col for correspondence search.')
    parser.add_argument('--reject', type=float, default=0.05,
                        help='Reject outliers with residual point distance larger than this value. '
                             'Disable with value <= 0.0')
    args = parser.parse_args()

    # load cfg from file
    if args.cfg_file is not None:
        for f in args.cfg_file:
            cfg_from_file(f)

    # load cfg from arguments
    if args.batch_size is not None:
        cfg_from_list(['DATASET.BATCH_SIZE', args.batch_size])
    if args.data_loader is not None:
        cfg_from_list(['DATALOADER_NUM', args.data_loader])
    if args.epoch is not None:
        cfg_from_list(['TRAIN.START_EPOCH', args.epoch, 'EVAL.EPOCH', args.epoch])
    if args.model is not None:
        cfg_from_list(['MODEL_NAME', args.model])
    if args.dataset is not None:
        cfg_from_list(['DATASET_PATH', args.dataset])
    if args.output is not None:
        cfg_from_list(['OUTPUT_PATH', args.output])
    if args.reg_iter is not None:
        cfg_from_list(['EVAL.ITERATION_NUM', args.reg_iter])
    if args.seed is not None:
        cfg_from_list(['RANDOM_SEED', args.seed])

    if args.output is None and len(cfg.MODEL_NAME) != 0 and len(cfg.DATASET_NAME) != 0:
        outp_path = get_output_dir(cfg.MODEL_NAME, cfg.DATASET_NAME + ('Unseen_' if cfg.DATASET.UNSEEN else 'Seen_') +
                                   ('PreW' if cfg.PRE_DCPWEIGHT else 'NoPreW') +
                                   str(cfg.PGM.FEATURES) + '_' + cfg.PGM.USEATTEND + '_' + cfg.DATASET.NOISE_TYPE)
        cfg_from_list(['OUTPUT_PATH', outp_path])
    elif args.output is not None:
        cfg_from_list(['OUTPUT_PATH', args.output])
    assert len(cfg.OUTPUT_PATH) != 0, 'Invalid OUTPUT_PATH! Make sure model name and dataset name are specified.'
    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)

    return args
