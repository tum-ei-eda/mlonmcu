"""Command line subcommand for the debug process."""


def get_parser(subparsers):
    """ "Define and return a subparser for the debug subcommand."""
    parser = subparsers.add_parser("debug", description="Debug model using ML on MCU flow.")
    parser.add_argument("-c", "--count")
    return parser
