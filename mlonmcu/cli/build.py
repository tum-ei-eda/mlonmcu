"""Command line subcommand for the build process."""

def get_build_parser(subparsers):
    """"Define and return a subparser for the build subcommand."""
    parser = subparsers.add_parser('build', description='Build model using the ML on MCU flow.')
    parser.add_argument('-c', '--count')
    return parser
