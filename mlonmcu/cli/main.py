"""Console script for mlonmcu."""
import argparse
import sys

from ..version import __version__

import mlonmcu.cli.init as init
#from .init import get_init_parser
import mlonmcu.cli.setup as setup
import mlonmcu.cli.flow as flow
import mlonmcu.cli.cleanup as cleanup
import mlonmcu.cli.check as check
import mlonmcu.cli.env as env
import mlonmcu.cli.models as models

import logging
# logging.basicConfig(format="[%(asctime)s]::%(pathname)s:%(lineno)d::%(levelname)s - %(message)s")
#logging.basicConfig(format="[%(asctime)s]::%(levelname)s - %(message)s")
logger = logging.getLogger("mlonmcu")
logger.setLevel(logging.DEBUG)

def main():
    """Console script for mlonmcu."""
    parser = argparse.ArgumentParser(description='ML on MCU Flow',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument('_', nargs='*')
    parser.add_argument('-V', '--version', action='version', version='mlonmcu ' + __version__)
    subparsers = parser.add_subparsers(dest="subcommand") # this line changed
    init_parser = init.get_parser(subparsers)
    setup_parser = setup.get_parser(subparsers)
    flow_parser = flow.get_parser(subparsers)
    # TODO: hide load,build,compile,run,debug,test behind flow subcommand?
    # trace_parser = get_trace_parser(subparsers)  # Handled as a flag to run subcommand or target-feature
    # TODO: cleanup
    cleanup_parser = cleanup.get_parser(subparsers)
    # TODO: check
    check_parser = check.get_parser(subparsers)
    # TODO: run
    # TODO: env
    env_parser = env.get_parser(subparsers)
    # TODO: models
    models_parser = models.get_parser(subparsers)
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
