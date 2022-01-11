"""Command line subcommand for installing a mlonmcu environment."""
import mlonmcu.setup.tasks
import mlonmcu.context
import mlonmcu.setup.install
import logging

from mlonmcu.cli.common import add_common_options, add_context_options

# logger = logging.getLogger(__name__)
logger = logging.getLogger("mlonmcu")
logger.setLevel(logging.DEBUG)


def get_parser(subparsers):
    """ "Define and return a subparser for the setup subcommand."""
    parser = subparsers.add_parser("setup", description="Setup ML on MCU dependencies.")
    parser.set_defaults(func=handle)
    parser.add_argument(
        "-p",
        "--progress",
        action="store_true",
        help="Display progress bar (default: %(default)s)",
    )
    add_context_options(parser)
    return parser


def handle(args):
    with mlonmcu.context.MlonMcuContext(path=args.home, lock=True) as context:
        # print(f"Environment: {context.environment}")
        progress = args.progress
        mlonmcu.setup.install.install_dependencies(context=context, progress=progress)
