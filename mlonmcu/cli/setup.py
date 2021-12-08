"""Command line subcommand for installing a mlonmcu environment."""
import mlonmcu.setup.tasks
import mlonmcu.context
import logging

#logger = logging.getLogger(__name__)
logger = logging.getLogger("mlonmcu")
logger.setLevel(logging.DEBUG)

def get_setup_parser(subparsers):
    """"Define and return a subparser for the setup subcommand."""
    parser = subparsers.add_parser('setup', description='Setup ML on MCU dependencies.')
    parser.set_defaults(func=handle)
    parser.add_argument('-H', '--home', type=str, default="", nargs=1, help="The path to the mlonmcu environment (overwriting $MLONMCU_HOME environment variable)")
    return parser

def handle(args):
    with mlonmcu.context.MlonMcuContext(lock=True) as context:
        print(f"Environment: {context.environment}")
        mlonmcu.setup.tasks.install_dependencies()
