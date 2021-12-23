"""Command line subcommand for checking the current environment."""

def get_parser(subparsers):
    """"Define and return a subparser for the check subcommand."""
    parser = subparsers.add_parser('check', description='Check ML on MCU environment.')
    return parser
