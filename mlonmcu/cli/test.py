"""Command line subcommand for the test process."""


def get_parser(subparsers):
    """ "Define and return a subparser for the test subcommand."""
    parser = subparsers.add_parser("test", description="Test model using ML on MCU flow.")
    parser.add_argument("-c", "--count")
    return parser
