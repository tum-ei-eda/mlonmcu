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
"""Command line subcommand for managing models."""

from mlonmcu.cli.common import add_common_options, add_context_options
from mlonmcu.context.context import MlonMcuContext
import mlonmcu.models


def add_models_options(parser):
    models_parser = parser.add_argument_group("models options")
    models_parser.add_argument(
        "--detailed",
        default=False,
        action="store_true",
        help="Display more information (default: %(default)s)",
    )


def get_parser(subparsers):
    """ "Define and return a subparser for the models subcommand."""
    parser = subparsers.add_parser("models", description="Manage ML on MCU models.")
    parser.set_defaults(func=handle)
    add_common_options(parser)
    add_context_options(parser)
    add_models_options(parser)
    return parser


def handle(args):
    with MlonMcuContext(path=args.home, deps_lock="read") as context:
        mlonmcu.models.print_summary(context, detailed=args.detailed)
