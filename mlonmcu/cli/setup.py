"""Command line subcommand for installing a mlonmcu environment."""
import mlonmcu.setup.tasks
import mlonmcu.context
import logging

from mlonmcu.cli.common import add_common_options, add_context_options

#logger = logging.getLogger(__name__)
logger = logging.getLogger("mlonmcu")
logger.setLevel(logging.DEBUG)

def get_setup_parser(subparsers):
    """"Define and return a subparser for the setup subcommand."""
    parser = subparsers.add_parser('setup', description='Setup ML on MCU dependencies.')
    parser.set_defaults(func=handle)
    add_context_options(parser)
    return parser

def handle(args):
    with mlonmcu.context.MlonMcuContext(path=args.home, lock=True) as context:
        # print(f"Environment: {context.environment}")
        mlonmcu.setup.tasks.install_dependencies()
