"""Command line subcommand for exporting session and runs."""
import mlonmcu.context

from mlonmcu.cli.common import (
    add_common_options,
    add_context_options,
    handle_logging_flags,
)


def add_export_options(parser):
    export_parser = parser.add_argument_group("export options")
    export_parser.add_argument("destination", help="Path to the output directory or archive")
    export_parser.add_argument(
        "-f",
        "--force",
        default=False,
        action="store_true",
        help="Overwrite files if the destination already exists (DANGEROUS)",
    )
    export_parser.add_argument(
        "-s",
        "--session",
        metavar="SESSION",
        type=int,
        nargs="?",
        default=None,
        const=-1,
        action="append",
        help="Which session(s) should be exported (default: latest session id)",
    )
    export_parser.add_argument(
        "-r",
        "--run",
        metavar="RUN",
        type=int,
        nargs="?",
        default=None,
        const=-1,
        action="append",
        help="Which run(s) should be exported (default: all runs of the selected session/latest run id)",
    )


def get_parser(subparsers):
    """ "Define and return a subparser for the cleanup subcommand."""
    parser = subparsers.add_parser("export", description="Export session/run artifacts to a directory/archive.")
    parser.set_defaults(func=handle)
    add_common_options(parser)
    add_context_options(parser)
    add_export_options(parser)
    return parser


def handle(args):
    with mlonmcu.context.MlonMcuContext(path=args.home, lock=True) as context:
        dest = args.destination
        interactive = not args.force
        sids = args.session
        rids = args.run
        context.export(dest, session_ids=sids, run_ids=rids, interactive=interactive)
