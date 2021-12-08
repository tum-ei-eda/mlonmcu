"""Command line subcommand for the run process."""

def get_run_parser(subparsers):
    """"Define and return a subparser for the run subcommand."""
    parser = subparsers.add_parser('run', description='Run model using ML on MCU flow.')
    parser.add_argument('-c', '--count')
    return parser
