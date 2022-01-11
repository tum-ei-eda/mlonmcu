"""Command line subcommand for the run process."""

import multiprocessing
import concurrent
import copy
import itertools
import logging

import mlonmcu
from mlonmcu.cli.common import add_common_options, add_context_options, add_flow_options
from mlonmcu.flow import SUPPORTED_FRAMEWORKS, SUPPORTED_FRAMEWORK_BACKENDS
from mlonmcu.cli.load import add_model_options, add_load_options
from mlonmcu.cli.build import add_build_options
from mlonmcu.cli.compile import (
    handle as handle_compile,
    add_compile_options,
    add_model_options,
)
from mlonmcu.flow.backend import Backend
from mlonmcu.flow.framework import Framework
from mlonmcu.session.run import RunStage

# rom mlonmcu.flow.tflite.framework import TFLiteFramework
# from mlonmcu.flow.tvm.framework import TVMFramework
# from mlonmcu.cli.compile import handle as handle_compile

logger = logging.getLogger("mlonmcu")
logger.setLevel(logging.DEBUG)


def add_run_options(parser):
    run_parser = parser.add_argument_group("run options")
    run_parser.add_argument(
        "--attach",
        action="store_true",
        help="Attach debugger to target (default: %(default)s)",
    )
    run_parser.add_argument(  # TODO: move to export?
        "--format",
        type=str,
        choices=["xlsx", "xls", "csv"],
        default="xlsx",
        help="Report file format (default: %(default)s)",
    )
    run_parser.add_argument(
        "--detailed-cycles",
        dest="detailed",
        action="store_true",
        help="Split total cycles to get the actual inference time as well as setup overhead (default: %(default)s)",
    )
    run_parser.add_argument(
        "--average-cycles",
        dest="average",
        action="store_true",
        help="Divide resulting cycles by the number of runs (default: %(default)s)",
    )


def get_parser(subparsers):
    """ "Define and return a subparser for the run subcommand."""
    parser = subparsers.add_parser(
        "run",
        description="Run model using ML on MCU flow. This is meant to reproduce the bahavior of the original `run.py` script in older versions of mlonmcu.",
    )
    parser.set_defaults(func=handle)
    add_model_options(parser)
    add_common_options(parser)
    add_context_options(parser)
    add_run_options(parser)
    add_compile_options(parser)
    add_build_options(parser)
    add_flow_options(parser)
    return parser


def check_args(context, args):
    print("CHECK ARGS")


def handle(args):
    print("HANLDE RUN")
    print(Framework.registry)
    # print(TFLiteFramework.backends)
    # print(TVMFramework.registry)
    print(args)
    with mlonmcu.context.MlonMcuContext(path=args.home, lock=True) as context:
        check_args(context, args)
        handle_compile(args, ctx=context)
        assert len(context.sessions) > 0
        session = context.sessions[-1]
        parallel = args.parallel
        progress = args.progress
        # session.process_runs(until=RunStage.RUN, per_stage=False, num_workers=parallel, progress=progress, context=context)
        session.process_runs(
            until=RunStage.RUN,
            per_stage=True,
            num_workers=parallel,
            progress=progress,
            context=context,
        )
        results = [run.result for run in session.runs]
        if args.detailed:

            def find_run_pairs(runs):
                return [
                    (a, b)
                    for a, b in itertools.permutations(runs, 2)
                    if a.model == b.model and a.backend == b.backend and a.num < b.num
                ]

            pairs = [
                (session.runs.index(a), session.runs.index(b))
                for a, b in find_run_pairs(session.runs)
            ]
            print("pairs", pairs)

    print("HANLDED RUN")
