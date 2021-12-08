"""Command line subcommand for manaing environments."""

def get_env_parser(subparsers):
    """"Define and return a subparser for the env subcommand."""
    parser = subparsers.add_parser('env', description='Manage ML on MCU environments.')
    parser.add_argument('-c', '--count')
    return parser
