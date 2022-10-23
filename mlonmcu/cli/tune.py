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
"""Command line subcommand for the tune stage."""

import mlonmcu
from mlonmcu.cli.common import kickoff_runs
from mlonmcu.cli.build import add_build_options, handle as handle_build
from mlonmcu.session.run import RunStage


def get_parser(subparsers, parent=None):
    """ "Define and return a subparser for the tune subcommand."""
    parser = subparsers.add_parser(
        "tune",
        description="Tune model using the ML on MCU flow.",
        parents=[parent] if parent else [],
        add_help=(parent is None),
    )
    parser.set_defaults(flow_func=handle)
    add_build_options(parser)
    return parser


def handle(args, ctx=None):
    if ctx:
        handle_build(args, ctx, require_target=True)
    else:
        # args.features.append("autotune")  # TODO: enable autotuning automatically?
        # args.features.append("autotuned")  # ?
        with mlonmcu.context.MlonMcuContext(path=args.home, env_lock="read", latest_session_link_lock = True) as context:
            handle_build(args, context, require_target=True)
            kickoff_runs(args, RunStage.TUNE, context)
