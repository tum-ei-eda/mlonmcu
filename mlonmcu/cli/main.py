"""Console script for mlonmcu."""
import argparse
import sys

from ..version import __version__

from .init import get_init_parser
from .setup import get_setup_parser
from .build import get_build_parser
from .debug import get_debug_parser
from .test import get_test_parser
from .trace import get_trace_parser
from .cleanup import get_cleanup_parser
from .check import get_check_parser
from .run import get_run_parser
from .env import get_env_parser
from .models import get_models_parser

import logging
logging.basicConfig(format="[%(asctime)s]::%(pathname)s:%(lineno)d::%(levelname)s - %(message)s")
logger = logging.getLogger("mlonmcu")
logger.setLevel(logging.DEBUG)

def main():
    """Console script for mlonmcu."""
    parser = argparse.ArgumentParser(description='ML on MCU Flow',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument('_', nargs='*')
    parser.add_argument('-V', '--version', action='version', version='%(prog)s ' + __version__)
    subparsers = parser.add_subparsers(dest="subcommand") # this line changed
    init_parser = get_init_parser(subparsers)
    setup_parser = get_setup_parser(subparsers)
    build_parser = get_build_parser(subparsers)
    debug_parser = get_debug_parser(subparsers)
    test_parser = get_test_parser(subparsers)
    trace_parser = get_trace_parser(subparsers)
    # TODO: cleanup
    cleanup_parser = get_cleanup_parser(subparsers)
    # TODO: check
    check_parser = get_check_parser(subparsers)
    # TODO: run
    run_parser = get_run_parser(subparsers)
    # TODO: env
    env_parser = get_env_parser(subparsers)
    # TODO: models
    models_parser = get_models_parser(subparsers)
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        raise RuntimeError("Invalid command. For usage details use '--help'!")

    #print("Arguments: " + str(args))
    #print("Replace this message by putting your code into "
    #      "mlonmcu.cli.main")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
