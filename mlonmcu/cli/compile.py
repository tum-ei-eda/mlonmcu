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
from mlonmcu.cli.build import (
    handle as handle_build,
    add_build_options,
)
from mlonmcu.context.context import MlonMcuContext
from mlonmcu.session.run import RunStage
from mlonmcu.platform.lookup import get_platforms_targets
from .helper.parse import extract_target_names, extract_platform_names, extract_config_and_feature_names


def add_compile_options(parser):
    add_build_options(parser)
    # compile_parser = parser.add_argument_group("compile options")


def get_parser(subparsers):
    """ "Define and return a subparser for the compile subcommand."""
    parser = subparsers.add_parser("compile", description="Compile model using ML on MCU flow.")
    parser.set_defaults(flow_func=handle)
    add_compile_options(parser)
    return parser


def _handle(args, context):
    handle_build(args, ctx=context)
    targets = extract_target_names(args, context=context)  # This will eventually be ignored below
    platforms = extract_platform_names(args, context=context)

    new_config, _, _, _ = extract_config_and_feature_names(args, context=context)
    platform_targets = get_platforms_targets(context, config=new_config)  # This will slow?

    assert len(context.sessions) > 0  # TODO: automatically request session if no active one is available
    session = context.sessions[-1]
    new_runs = []
    for run in session.runs:
        if run.target is None:
            # assert run.compile_platform is None
            targets_ = targets
        else:
            targets_ = [None]
        for target_name in targets_:
            new_run = run.copy()
            if target_name is not None:
                platform_name = None
                for platform in platforms:
                    candidates = platform_targets[platform]
                    if target_name in candidates:
                        platform_name = platform
                new_run.add_platform_by_name(platform_name, context=context)
                new_run.add_target_by_name(target_name, context=context)
            new_runs.append(new_run)
    session.runs = new_runs


def check_args(context, args):
    # print("CHECK ARGS")
    pass


def handle(args, ctx=None):
    if ctx:
        _handle(args, ctx)
    else:
        with MlonMcuContext(path=args.home, deps_lock="read") as context:
            _handle(args, context)
            kickoff_runs(args, RunStage.COMPILE, context)
