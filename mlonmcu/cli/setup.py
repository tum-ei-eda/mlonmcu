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
"""Command line subcommand for installing a mlonmcu environment."""
import mlonmcu.setup.tasks
import mlonmcu.context
from mlonmcu.setup import setup

from mlonmcu.cli.common import (
    add_common_options,
    add_context_options,
    handle_logging_flags,
)
from .helper.parse import extract_config_and_init_features


def add_setup_options(parser):
    setup_parser = parser.add_argument_group("setup options")
    setup_parser.add_argument(
        "-f",
        "--feature",
        type=str,
        metavar="FEATURE",
        # nargs=1,
        action="append",
        choices=[],  # FIXME: only setup features?
        help="Enabled features for setup (choices: %(choices)s)",
    )
    setup_parser.add_argument(
        "-c",
        "--config",
        metavar="KEY=VALUE",
        nargs="+",
        action="append",
        help="Set a number of key-value pairs "
        "(do not put spaces before or after the = sign). "
        "If a value contains spaces, you should define "
        "it with double quotes: "
        'foo="this is a sentence". Note that '
        "values are always treated as strings.",
    )
    setup_parser.add_argument(
        "-p",
        "--progress",
        action="store_true",
        help="Display progress bar (default: %(default)s)",
    )
    setup_parser.add_argument(
        "-r",
        "--rebuild",
        action="store_true",
        help="Trigger a rebuild/refresh of already installed dependencies (default: %(default)s)",
    )
    setup_parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="Only print a list of the tasks to be processed and quit (default: %(default)s)",
    )
    setup_parser.add_argument(
        "--task",
        type=str,
        nargs=1,
        default=None,
        help="Invoke a single task manually by name (default: %(default)s)",
    )


def get_parser(subparsers):
    """ "Define and return a subparser for the setup subcommand."""
    parser = subparsers.add_parser("setup", description="Setup ML on MCU dependencies.")
    parser.set_defaults(func=handle)
    add_common_options(parser)
    add_context_options(parser)
    add_setup_options(parser)
    return parser


def handle(args):
    with mlonmcu.context.MlonMcuContext(path=args.home, lock=True) as context:
        config, features = extract_config_and_init_features(args)
        installer = setup.Setup(features=features, config=config, context=context)
        if args.list:
            order = installer.get_dependency_order()
            print("The following tasks have been selected:")
            print("\n".join(["- " + key for key in order]))
        elif args.task:
            installer.invoke_single_task(
                args.task[0],
                progress=args.progress,
                rebuild=args.rebuild,
            )
        else:
            installer.install_dependencies(
                progress=args.progress,
                rebuild=args.rebuild,
            )
