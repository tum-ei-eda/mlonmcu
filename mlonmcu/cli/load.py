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
"""Command line subcommand for the load stage."""

from mlonmcu.cli.common import (
    add_common_options,
    add_context_options,
    add_model_options,
    add_flow_options,
    kickoff_runs,
)

from .helper.parse import extract_config_and_feature_names, extract_frontend_names, extract_postprocess_names
from mlonmcu.context.context import MlonMcuContext
from mlonmcu.models import SUPPORTED_FRONTENDS
from mlonmcu.models.lookup import apply_modelgroups
from mlonmcu.session.run import RunStage


def add_load_options(parser):
    add_common_options(parser)
    add_context_options(parser)
    add_flow_options(parser)
    add_model_options(parser)
    load_parser = parser.add_argument_group("load options")
    load_parser.add_argument(
        "--frontend",
        type=str,
        metavar="FRONTEND",
        choices=SUPPORTED_FRONTENDS.keys(),
        default=None,
        nargs=1,
        help="Explicitly choose the frontends to use (choices: %(choices)s)",
    )


def get_parser(subparsers):
    """ "Define and return a subparser for the load subcommand."""
    parser = subparsers.add_parser("load", description="Load model using the ML on MCU flow.")
    parser.set_defaults(flow_func=handle)
    add_load_options(parser)
    return parser


def _handle(args, context):
    config = context.environment.vars
    new_config, features, gen_config, gen_features = extract_config_and_feature_names(args, context=context)
    config.update(new_config)
    frontends = extract_frontend_names(args, context=context)
    postprocesses = extract_postprocess_names(args, context=context)
    session = context.get_session(label=args.label, resume=args.resume, config=config)
    models = apply_modelgroups(args.models, context=context)
    for model in models:
        for f in gen_features:
            for c in gen_config:
                all_config = {**config, **c}
                run = session.create_run(config=all_config)
                all_features = list(set(features + f))
                run.add_features_by_name(all_features, context=context)  # TODO do this before load.py?
                run.add_frontends_by_name(frontends, context=context)
                run.add_model_by_name(model, context=context)
                run.add_postprocesses_by_name(postprocesses, context=context)  # TODO do this before load.py?


def handle(args, ctx=None):
    if ctx:
        _handle(args, ctx)
    else:
        with MlonMcuContext(path=args.home, deps_lock="read") as context:
            _handle(args, context)
            kickoff_runs(args, RunStage.LOAD, context)
