"""Command line subcommand for managing models."""

from mlonmcu.cli.common import add_common_options, add_context_options
import mlonmcu.models


def add_models_options(parser):
    models_parser = parser.add_argument_group("models options")
    models_parser.add_argument(
        "--detailed",
        default=False,
        action="store_true",
        help="Display more information (default: %(default)s)",
    )


def get_parser(subparsers):
    """ "Define and return a subparser for the models subcommand."""
    parser = subparsers.add_parser("models", description="Manage ML on MCU models.")
    parser.set_defaults(func=handle)
    add_common_options(parser)
    add_context_options(parser)
    add_models_options(parser)
    return parser


def handle(args):
    with mlonmcu.context.MlonMcuContext(path=args.home, lock=True) as context:
        mlonmcu.models.print_summary(context, detailed=args.detailed)
