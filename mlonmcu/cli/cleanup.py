"""Command line subcommand for cleaning up the current environment."""

def get_cleanup_parser(subparsers):
    """"Define and return a subparser for the cleanup subcommand."""
    parser = subparsers.add_parser('cleanup', description='Cleanup ML on MCU environment.')
    return parser
