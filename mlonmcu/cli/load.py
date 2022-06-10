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

import mlonmcu
from mlonmcu.cli.common import (
    add_common_options,
    add_context_options,
    add_model_options,
    add_flow_options,
    kickoff_runs,
)
from mlonmcu.config import resolve_required_config

# from .helper.parse import extract_config_and_init_features
from .helper.parse import extract_config_and_feature_names, extract_frontend_names, extract_postprocess_names
from mlonmcu.models import SUPPORTED_FRONTENDS
from mlonmcu.models.lookup import lookup_models, map_frontend_to_model
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


# def init_frontends(frontend_names, features, config, context=None):
#     names = []
#     if isinstance(frontend_names, list) and len(frontend_names) > 0:
#         names = frontend_names
#     elif isinstance(frontend_names, str):
#         names = [frontend_names]
#     else:
#         # No need to specify a default, because we just use the provided order in the environment.yml
#         assert frontend_names is None, "TODO"
#         assert context is not None, "Need context to resolve default frontends"
#         all_frontend_names = context.environment.lookup_frontend_configs(names_only=True)
#         names.extend(all_frontend_names)
#     frontends = []
#     for frontend_name in names:
#         frontend_cls = SUPPORTED_FRONTENDS[frontend_name]
#         required_keys = frontend_cls.REQUIRED
#         frontend_config = config.copy()
#         frontend_config.update(
#             resolve_required_config(
#                 required_keys,
#                 features=features,
#                 config=config,
#                 cache=context.cache if context else None,
#             )
#         )
#         try:
#             frontend = frontend_cls(features=features, config=frontend_config)
#         # except Exception as err:
#         except Exception:
#             # raise RuntimeError() from err  # Comment this in to debug frontend issues
#             print("Frontend could not be initialized. Continuing with next one...")
#             continue
#         frontends.append(frontend)
#     assert len(frontends) > 0, "Could not initialize at least one frontend for the given set of features"
#     return frontends


def _handle(args, context):
    # model_names = args.models
    # new_config, features = extract_config_and_init_features(args, context=context)
    config = context.environment.vars
    new_config, features = extract_config_and_feature_names(args, context=context)
    config.update(new_config)
    frontends = extract_frontend_names(args, context=context)
    postprocesses = extract_postprocess_names(args, context=context)
    postprocesses = list(set(args.postprocess)) if args.postprocess is not None else []
    # frontends = init_frontends(args.frontend, features=features, config=config, context=context)
    # model_hints = lookup_models(model_names, frontends=frontends, context=context)
    session = context.get_session(label=args.label, resume=args.resume, config=config)
    # for hint in model_hints:
    for model in args.models:
        # model, frontend = map_frontend_to_model(
        #     hint, frontends, backend=None
        # )  # TODO: we do not yet know the backend...
        # run = session.create_run(model=model, features=features, config=config)
        run = session.create_run(config=config)
        run.add_features_by_name(features, context=context)  # TODO do this before load.py?
        run.add_frontends_by_name(frontends, context=context)
        run.add_model_by_name(model, context=context)
        run.add_postprocesses_by_name(postprocesses, context=context)  # TODO do this before load.py?


def handle(args, ctx=None):
    if ctx:
        _handle(args, ctx)
    else:
        with mlonmcu.context.MlonMcuContext(path=args.home, lock=True) as context:
            _handle(args, context)
            kickoff_runs(args, RunStage.LOAD, context)
