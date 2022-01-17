"""Command line subcommand for installing a mlonmcu environment."""
import mlonmcu.setup.tasks
import mlonmcu.context
import mlonmcu.setup.install

from mlonmcu.cli.common import (
    add_common_options,
    add_context_options,
    handle_logging_flags,
)


def get_parser(subparsers):
    """ "Define and return a subparser for the setup subcommand."""
    parser = subparsers.add_parser("setup", description="Setup ML on MCU dependencies.")
    parser.set_defaults(func=handle)
    add_common_options(parser)
    add_context_options(parser)
    parser.add_argument(
        "-p",
        "--progress",
        action="store_true",
        help="Display progress bar (default: %(default)s)",
    )
    return parser


def handle(args):
    with mlonmcu.context.MlonMcuContext(path=args.home, lock=True) as context:
        # print(f"Environment: {context.environment}")
        progress = args.progress
        mlonmcu.setup.install.install_dependencies(context=context, progress=progress)
