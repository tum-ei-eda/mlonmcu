"""Command line subcommand for managing models."""

import mlonmcu.models

def get_models_parser(subparsers):
    """"Define and return a subparser for the models subcommand."""
    parser = subparsers.add_parser('models', description='Manage ML on MCU models.')
    parser.set_defaults(func=handle)
    parser.add_argument('-H', '--home', type=str, default="", nargs=1, help="The path to the mlonmcu environment (overwriting $MLONMCU_HOME environment variable)")
    parser.add_argument('--detailed', default=False, action="store_true", help="Display more information (default: %(default)s)")
    return parser

def handle(args):
    with mlonmcu.context.MlonMcuContext(lock=True) as context:
        mlonmcu.models.print_summary(context, detailed=args.detailed)
