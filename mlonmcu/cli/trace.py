"""Command line subcommand for the trace process."""


def get_trace_parser(subparsers):
    """ "Define and return a subparser for the trace subcommand."""
    parser = subparsers.add_parser("trace", description="Trace model ML on MCU flow.")
    parser.add_argument("-c", "--count")
    return parser
