"""Command line subcommand for invoking the mlonmcu flow."""

import sys

import multiprocessing

import mlonmcu.cli.load as load
import mlonmcu.cli.tune as tune
import mlonmcu.cli.build as build
import mlonmcu.cli.compile as compile_  # compile is a builtin name
import mlonmcu.cli.debug as debug
import mlonmcu.cli.test as test
import mlonmcu.cli.run as run
from mlonmcu.cli.common import add_flow_options, add_common_options

# from .trace import get_trace_parser


def get_parser(subparsers, parent=None):
    """ "Define and return a subparser for the flow subcommand."""
    parser = subparsers.add_parser(
        "flow",
        description="Invoke ML on MCU flow",
        parents=[parent] if parent else [],
        add_help=(parent is None),
    )
    parser.set_defaults(func=handle)
    add_common_options(parser)
    add_flow_options(parser)
    subparsers = parser.add_subparsers(dest="subcommand2")  # this line changed
    load_parser = load.get_parser(subparsers)
    tune_parser = tune.get_parser(subparsers)
    build_parser = build.get_parser(subparsers)
    compile_parser = compile_.get_parser(subparsers)
    run_parser = run.get_parser(subparsers)
    debug_parser = debug.get_parser(subparsers)
    test_parser = test.get_parser(subparsers)


def handle(args):
    """Callback function which will be called to process the flow subcommand"""
    if hasattr(args, "flow_func"):
        args.flow_func(args)
    else:
        print("Invalid command. Check 'mlonmcu flow --help' for the available subcommands!")
        sys.exit(1)
