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
"""Command line subcommand for initializing a mlonmcu environment."""

import os

from mlonmcu.environment.templates import get_template_names
from mlonmcu.environment.config import get_environments_dir, DEFAULTS
from mlonmcu.environment.init import initialize_environment
from .helper.parse import extract_config


def add_init_options(parser):
    init_parser = parser.add_argument_group("init options")
    init_parser.add_argument(
        "-n",
        "--name",
        metavar="NAME",
        nargs=1,
        type=str,
        default="",
        help="Environment name (default: %(default)s)",
    )
    init_parser.add_argument(
        "-t",
        "--template",
        metavar="TEMPLATE",
        nargs=1,
        type=str,
        # choices=get_template_names(),
        default=DEFAULTS["template"],
        help="Environment template name or path (default: %(default)s, available: "
        + ", ".join(get_template_names())
        + ")",
    )
    init_parser.add_argument(
        "DIR",
        nargs="?",
        type=str,
        default=get_environments_dir(),
        help="Environment directory (default: " + os.path.join(get_environments_dir(), "{NAME}") + ")",
    )
    init_parser.add_argument(
        "--non-interactive",
        dest="non_interactive",
        action="store_true",
        help="Do not ask questions interactively",
    )
    init_parser.add_argument(
        "--venv",
        default=None,
        action="store_true",
        help="Create virtual python environment",
    )
    init_parser.add_argument(
        "--clone-models",
        dest="clone_models",
        default=None,
        action="store_true",
        help="Clone models directory into environment",
    )
    init_parser.add_argument(
        "--register",
        default=None,
        action="store_true",
        help="Add environment to the list of environment for the current user",
    )
    init_parser.add_argument(
        "--allow-exists",
        dest="allow_exists",
        default=None,
        action="store_true",
        help="Allow overwriting an existing environment directory",
    )
    init_parser.add_argument(
        "-c",
        "--config",
        metavar="KEY=VALUE",
        nargs="+",
        action="append",
        help=(
            "Set a number of key-value pairs "
            "(do not put spaces before or after the = sign). "
            "If a value contains spaces, you should define "
            "it with double quotes: "
            'foo="this is a sentence". Note that '
            "values are always treated as strings."
        ),
    )


def get_parser(subparsers):
    """ "Define and return a subparser for the init subcommand."""
    parser = subparsers.add_parser("init", description="Initialize ML on MCU environment.")
    parser.set_defaults(func=handle)
    add_init_options(parser)
    return parser


def handle(args):
    """Callback function which will be called to process the init subcommand"""
    name = args.name[0] if isinstance(args.name, list) else args.name
    config, _ = extract_config(args)
    initialize_environment(
        args.DIR,
        name,
        create_venv=args.venv,
        interactive=not args.non_interactive,
        clone_models=args.clone_models,
        register=args.register,
        template=args.template[0] if isinstance(args.template, list) else args.template,
        allow_exists=args.allow_exists,
        config=config,
    )
