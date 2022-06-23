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
"""Command line subcommand for the build process."""

import mlonmcu
from mlonmcu.flow import get_available_backend_names
from mlonmcu.cli.common import kickoff_runs
from mlonmcu.cli.load import handle as handle_load, add_load_options
from mlonmcu.session.run import RunStage
from mlonmcu.platform.lookup import get_platforms_targets
from .helper.parse import extract_backend_names, extract_target_names, extract_platform_names


def add_build_options(parser):
    # TODO: rename to build_group
    add_load_options(parser)
    build_parser = parser.add_argument_group("build options")
    build_parser.add_argument(
        "-b",
        "--backend",
        type=str,
        action="append",
        choices=get_available_backend_names(),
        help="Backends to use (default: %(default)s)",
    )


def get_parser(subparsers, parent=None):
    """ "Define and return a subparser for the build subcommand."""
    parser = subparsers.add_parser(
        "build",
        description="Build model using the ML on MCU flow.",
        parents=[parent] if parent else [],
        add_help=(parent is None),
    )
    parser.set_defaults(flow_func=handle)
    add_build_options(parser)
    return parser


def _handle(args, context):
    handle_load(args, ctx=context)
    backends = extract_backend_names(args, context=context)
    targets = extract_target_names(args, context=None)
    platforms = extract_platform_names(args, context=context)

    platform_targets = get_platforms_targets(context)  # This will be slow?

    assert len(context.sessions) > 0
    session = context.sessions[-1]
    new_runs = []
    for run in session.runs:
        for target_name in targets:
            for backend_name in backends:
                new_run = run.copy()
                if target_name is not None:
                    platform_name = None
                    for platform in platforms:
                        candidates = platform_targets[platform]
                        if target_name in candidates:
                            platform_name = platform
                    assert (
                        platform_name is not None
                    ), f"Unable to find a suitable platform for the target '{target_name}'"
                    new_run.add_platform_by_name(platform_name, context=context)
                    new_run.add_target_by_name(target_name, context=context)
                new_run.add_backend_by_name(backend_name, context=context)
                new_runs.append(new_run)

    session.runs = new_runs


def handle(args, ctx=None):
    if ctx:
        _handle(args, ctx)
    else:
        with mlonmcu.context.MlonMcuContext(path=args.home, lock=True) as context:
            _handle(args, context)
            kickoff_runs(args, RunStage.BUILD, context)
