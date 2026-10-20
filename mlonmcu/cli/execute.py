#
# Copyright (c) 2026 TUM Department of Electrical and Computer Engineering.
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
"""Execute a configured pipeline through a built-in or user-defined stage."""

from mlonmcu.cli.common import kickoff_runs
from mlonmcu.cli.compile import handle as handle_compile
from mlonmcu.cli.run import add_run_options
from mlonmcu.context.context import MlonMcuContext
from mlonmcu.session.stage import stage_key


def get_parser(subparsers):
    parser = subparsers.add_parser("execute", description=__doc__)
    parser.set_defaults(flow_func=handle)
    add_run_options(parser)
    parser.add_argument("--until", default="done", help="Inclusive pipeline stage ID (default: done)")
    return parser


def handle(args):
    with MlonMcuContext(path=args.home, deps_lock="read") as context:
        handle_compile(args, context)
        kickoff_runs(args, stage_key(args.until), context)
