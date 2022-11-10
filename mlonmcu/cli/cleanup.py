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
"""Command line subcommand for cleaning up the current environment."""
from mlonmcu.context.context import MlonMcuContext
from mlonmcu.setup import setup

from mlonmcu.cli.common import (
    add_common_options,
    add_context_options,
)


def get_parser(subparsers):
    """ "Define and return a subparser for the cleanup subcommand."""
    parser = subparsers.add_parser("cleanup", description="Cleanup ML on MCU environment.")
    parser.set_defaults(func=handle)
    add_common_options(parser)
    add_context_options(parser)
    parser.add_argument(
        "-f",
        "--force",
        default=False,
        action="store_true",
        help="Do not ask before removing disk contents (DANGEROUS)",
    )
    parser.add_argument(
        "-k",
        "--keep",
        metavar="KEEP",
        type=int,
        default=10,
        help="Remove everything except the latest KEEP sessions (default: %(default)s)",
    )
    parser.add_argument(
        "--deps",
        default=False,
        action="store_true",
        help="Also delete all dependencies from the environment.",
    )
    parser.add_argument(
        "--cache",
        default=False,
        action="store_true",
        help="Clear the environments dependency cache.",
    )
    return parser


def handle(args):
    with MlonMcuContext(path=args.home, deps_lock="write") as context:
        interactive = not args.force
        keep = args.keep
        context.cleanup_sessions(keep=keep, interactive=interactive)
        installer = setup.Setup(context=context)
        if args.deps:
            # This will also remove the cache file
            installer.clean_dependencies(interactive=interactive)
        elif args.cache:
            installer.clean_cache(interactive=interactive)
