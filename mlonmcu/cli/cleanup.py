"""Command line subcommand for cleaning up the current environment."""
import mlonmcu.context
from mlonmcu.setup import setup

from mlonmcu.cli.common import (
    add_common_options,
    add_context_options,
    handle_logging_flags,
)


def get_parser(subparsers):
    """ "Define and return a subparser for the cleanup subcommand."""
    parser = subparsers.add_parser("cleanup", description="Cleanup ML on MCU environment.")
    parser.set_defaults(func=handle)
    add_common_options(parser)
    add_context_options(parser)
    parser.add_argument(
        "-f",
        "--force",
        default=False,
        action="store_true",
        help="Do not ask before removing disk contents (DANGEROUS)",
    )
    parser.add_argument(
        "-k",
        "--keep",
        metavar="KEEP",
        type=int,
        default=10,
        help="Remove everything except the latest KEEP sessions (default: %(default)s)",
    )
    parser.add_argument(
        "--deps",
        default=False,
        action="store_true",
        help="Also delete all dependencies from the environment.",
    )
    parser.add_argument(
        "--cache",
        default=False,
        action="store_true",
        help="Clear the environments dependency cache.",
    )
    return parser


def handle(args):
    with mlonmcu.context.MlonMcuContext(path=args.home, lock=True) as context:
        interactive = not args.force
        keep = args.keep
        context.cleanup_sessions(keep=keep, interactive=interactive)
        installer = setup.Setup(context=context)
        if args.deps:
            # This will also remove the cache file
            installer.clean_dependencies(interactive=interactive)
        elif args.cache:
            installer.clean_cache(interactive=interactive)
