#
# Copyright (c) 2022 TUM Department of Electrical and Computer Engineering.
#
# This file is part of MLonMCU.
# See https://github.com/tum-ei-eda/mlonmcu.git for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Command line subcommand for the run process."""

import multiprocessing
import concurrent
import copy
import itertools

import mlonmcu
from mlonmcu.cli.common import add_common_options, add_context_options, kickoff_runs
from mlonmcu.flow import SUPPORTED_FRAMEWORKS, SUPPORTED_FRAMEWORK_BACKENDS
from mlonmcu.target import SUPPORTED_TARGETS
from mlonmcu.cli.build import (
    handle as handle_build,
    add_build_options,
)
from mlonmcu.config import resolve_required_config
from mlonmcu.flow.backend import Backend
from mlonmcu.flow.framework import Framework
from mlonmcu.session.run import RunStage


def add_compile_options(parser):
    add_build_options(parser)
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


def get_parser(subparsers):
    """ "Define and return a subparser for the compile subcommand."""
    parser = subparsers.add_parser("compile", description="Compile model using ML on MCU flow.")
    parser.set_defaults(flow_func=handle)
    add_compile_options(parser)
    return parser


def _handle(args, context):
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
    assert len(context.sessions) > 0  # TODO: automatically request session if no active one is available
    session = context.sessions[-1]
    new_runs = []
    for run in session.runs:
        for target_name in targets:
            for n in num:
                new_run = run.copy()
                if args.platform:  # TODO: move this somewhere else
                    platform_name = args.platform[0]
                else:
                    platform_name = "mlif"
                new_run.add_target_by_name(target_name, context=context)
                new_run.add_platform_by_name(platform_name, context=context)  # TODO: do this implicitly
                new_run.debug = debug
                new_run.num = n
                new_runs.append(new_run)
    session.runs = new_runs


def check_args(context, args):
    # print("CHECK ARGS")
    pass


def handle(args, ctx=None):
    if ctx:
        _handle(args, ctx)
    else:
        with mlonmcu.context.MlonMcuContext(path=args.home, lock=True) as context:
            _handle(args, context)
            kickoff_runs(args, RunStage.COMPILE, context)
