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
"""Command line subcommand for invoking the mlonmcu flow."""

import sys

from mlonmcu.context.context import MlonMcuContext
import mlonmcu.platform.lookup
import mlonmcu.cli.load as load
import mlonmcu.cli.tune as tune
import mlonmcu.cli.build as build
import mlonmcu.cli.compile as compile_  # compile is a builtin name
import mlonmcu.cli.run as run
from mlonmcu.cli.common import add_flow_options, add_common_options, add_context_options


def get_parser(subparsers, parent=None):
    """ "Define and return a subparser for the flow subcommand."""
    parser = subparsers.add_parser(
        "flow",
        description="Invoke ML on MCU flow",
        parents=[parent] if parent else [],
        add_help=(parent is None),
    )
    parser.set_defaults(func=handle)
    add_common_options(parser)
    add_context_options(parser)
    add_flow_options(parser)
    subparsers = parser.add_subparsers(dest="subcommand2")  # this line changed
    load.get_parser(subparsers)
    tune.get_parser(subparsers)
    build.get_parser(subparsers)
    compile_.get_parser(subparsers)
    run.get_parser(subparsers)


def handle_list_targets(args):
    with MlonMcuContext(path=args.home, deps_lock="read") as context:
        mlonmcu.platform.lookup.print_summary(context=context)


def handle(args):
    """Callback function which will be called to process the flow subcommand"""
    if args.list_targets:
        handle_list_targets(args)
    else:
        if hasattr(args, "flow_func"):
            args.flow_func(args)
        else:
            print("Invalid command. Check 'mlonmcu flow --help' for the available subcommands!")
            sys.exit(1)
