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

import copy

import mlonmcu
from mlonmcu.flow import get_available_backend_names
import mlonmcu.flow.tflite
import mlonmcu.flow.tvm
from mlonmcu.models.model import Model
from mlonmcu.session.run import Run
from mlonmcu.session.session import Session
from mlonmcu.cli.common import (
    add_common_options,
    add_context_options,
    add_model_options,
    add_flow_options,
    kickoff_runs,
)
from mlonmcu.config import resolve_required_config
from mlonmcu.cli.load import handle as handle_load, add_load_options
from mlonmcu.flow import SUPPORTED_BACKENDS, SUPPORTED_FRAMEWORKS
from mlonmcu.session.run import RunStage


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
    backend_names = args.backend
    if isinstance(backend_names, list) and len(backend_names) > 0:
        backends = backend_names
    elif isinstance(backend_names, str):
        backends = [backend_names]
    else:
        assert backend_names is None, "TODO"
        frameworks = context.environment.get_default_frameworks()
        backends = []
        for framework in frameworks:
            framework_backends = context.environment.get_default_backends(framework)
            backends.extend(framework_backends)
    assert len(context.sessions) > 0
    session = context.sessions[-1]
    new_runs = []
    for run in session.runs:
        for backend_name in backends:
            new_run = run.copy()
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
