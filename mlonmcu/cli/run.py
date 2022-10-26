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

from mlonmcu.cli.common import kickoff_runs
from mlonmcu.cli.compile import (
    handle as handle_compile,
    add_compile_options,
)
from mlonmcu.context.context import MlonMcuContext
from mlonmcu.session.run import RunStage


def add_run_options(parser):
    add_compile_options(parser)
    _ = parser.add_argument_group("run options")


def get_parser(subparsers):
    """ "Define and return a subparser for the run subcommand."""
    parser = subparsers.add_parser(
        "run",
        description="Run model using ML on MCU flow. This is meant to reproduce the behavior"
        + " of the original `run.py` script in older versions of mlonmcu.",
    )
    parser.set_defaults(flow_func=handle)
    add_run_options(parser)
    return parser


def check_args(context, args):
    # print("CHECK ARGS")
    pass


def handle(args):
    with MlonMcuContext(path=args.home, deps_lock="read") as context:
        check_args(context, args)
        handle_compile(args, context)
        kickoff_runs(args, RunStage.RUN, context)
