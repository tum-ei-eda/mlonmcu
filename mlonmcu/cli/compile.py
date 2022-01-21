"""Command line subcommand for the run process."""

import multiprocessing
import concurrent
import copy
import itertools

import mlonmcu
from mlonmcu.cli.common import add_common_options, add_context_options
from mlonmcu.flow import SUPPORTED_FRAMEWORKS, SUPPORTED_FRAMEWORK_BACKENDS
from mlonmcu.target import SUPPORTED_TARGETS
from mlonmcu.cli.build import (
    handle as handle_build,
    add_build_options,
    add_model_options,
)
from mlonmcu.config import resolve_required_config
from mlonmcu.flow.backend import Backend
from mlonmcu.flow.framework import Framework
from mlonmcu.session.run import RunStage


def add_compile_options(parser):
    compile_parser = parser.add_argument_group("compile options")
    compile_parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Build target sorftware in DEBUG mode (default: %(default)s)",
    )
    compile_parser.add_argument(
        "--num",
        action="append",
        type=int,
        help="Number of runs in simulation (default: %(default)s)",
    )
    compile_parser.add_argument(
        "--ignore-data",
        dest="ignore_data",
        action="store_true",
        help="Do not use MLIF inout data in debug mode (default: %(default)s)",
    )


def get_parser(subparsers):
    """ "Define and return a subparser for the compile subcommand."""
    parser = subparsers.add_parser(
        "compile", description="Compile model using ML on MCU flow."
    )
    parser.set_defaults(func=handle)
    add_model_options(parser)
    add_common_options(parser)
    add_context_options(parser)
    add_compile_options(parser)
    add_build_options(parser)
    return parser


def _handle(context, args):
    handle_build(args, ctx=context)
    num = args.num if args.num else [1]
    if isinstance(args.target, list) and len(args.target) > 0:
        targets = args.target
    elif isinstance(args.target, str):
        targets = [args.target]
    else:
        assert args.target is None, "TODO"
        targets = context.environment.get_default_targets()
        assert len(targets) > 0, "TODO"

    debug = args.debug
    assert (
        len(context.sessions) > 0
    )  # TODO: automatically request session if no active one is available
    session = context.sessions[-1]
    new_runs = []
    for run in session.runs:
        for target_name in targets:
            assert target_name in SUPPORTED_TARGETS, "TODO"
            for n in num:
                new_run = copy.deepcopy(run)
                target_cls = SUPPORTED_TARGETS[target_name]
                required_keys = target_cls.REQUIRED
                cache_flags = get_cache_flags(new_run.features)
                for key in required_keys:
                    if key not in new_run.config:
                        flags = cache_flags.get(key, ())
                        if len(context.cache) == 0:
                            raise RuntimeError(
                                "The dependency cache is empty! Make sure `to run `mlonmcu` setup first.`"
                            )
                        if (key, flags) in context.cache:
                            value = context.cache[key, flags]
                            new_run.config[key] = value
                        else:
                            raise RuntimeError(
                                "Dependency cache miss for required key '{key}'. Try re-running `mlonmcu setup`."
                            )
                target_inst = target_cls(
                    features=new_run.features, config=new_run.config
                )
                new_run.target = target_inst
                new_run.debug = debug
                new_run.num = n
                new_run.num = n
                new_runs.append(new_run)
    session.runs = new_runs
    for run in session.runs:
        run.compile(context=context)


def check_args(context, args):
    # print("CHECK ARGS")
    pass


def handle(args, ctx=None):
    if ctx:
        _handle(ctx, args)
    else:
        with mlonmcu.context.MlonMcuContext(path=args.home, lock=True) as context:
            _handle(context, args)
