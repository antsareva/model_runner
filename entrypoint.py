import argparse
from model.model import run
from conf.conf import settings
import logging


def parse_args():
    parser = argparse.ArgumentParser(
        prog='Model Runner',
        description='Runs ML models')

    parser.add_argument('--model', type=str, required=False, default='linear_regression',
                        choices=['linear_regression', 'random_forest'],
                        help='Path to model file')

    parser.add_argument('--dumped_model_path', type=str, required=False,
                        help='Path to dumped_model')

    parser.add_argument('--dump_model_path', type=str, required=False,
                        help='Path to file to dump model')

    parser.add_argument('--max_depth', type=int, required=False, default=settings.RF_MAX_DEPTH,
                        help='Max depth for random forest')

    parser.add_argument('--test-size', type=float, required=False, default=settings.TEST_SIZE,
                        help='Test size')

    parser.add_argument('--log-level', type=str, required=False, default=settings.LOG_LEVEL,
                        help='Log level')

    return parser.parse_args()


def main():
    args = parse_args()
    if args.log_level is not None:
        logging.getLogger().setLevel(args.log_level)

    run(args.dumped_model_path, args.model, args.dump_model_path, args.max_depth, args.test_size)


main()
