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
"""Command line subcommand for exporting session and runs."""
from mlonmcu.context.context import MlonMcuContext

from mlonmcu.cli.common import (
    add_common_options,
    add_context_options,
)


def add_export_options(parser):
    export_parser = parser.add_argument_group("export options")
    export_parser.add_argument(
        "destination",
        nargs="?",
        default="exported.zip",
        help="Path to the output directory or archive (default: %(default)s)",
    )
    export_parser.add_argument(
        "-f",
        "--force",
        default=False,
        action="store_true",
        help="Overwrite files if the destination already exists (DANGEROUS)",
    )
    export_parser.add_argument(
        "-l",
        "--list",
        default=False,
        action="store_true",
        help="Print a summary of all available sessions and runs",
    )
    export_parser.add_argument(
        "-s",
        "--session",
        metavar="SESSION",
        type=int,
        nargs="?",
        default=None,
        const=-1,
        action="append",
        help="Which session(s) should be exported (default: latest session id)",
    )
    export_parser.add_argument(
        "-r",
        "--run",
        metavar="RUN",
        type=int,
        nargs="?",
        default=None,
        const=-1,
        action="append",
        help="Which run(s) should be exported (default: all runs of the selected session/latest run id)",
    )


def get_parser(subparsers):
    """ "Define and return a subparser for the cleanup subcommand."""
    parser = subparsers.add_parser("export", description="Export session/run artifacts to a directory/archive.")
    parser.set_defaults(func=handle)
    add_common_options(parser)
    add_context_options(parser)
    add_export_options(parser)
    return parser


def handle(args):
    with MlonMcuContext(path=args.home, deps_lock="read") as context:
        if args.list:
            context.print_summary(sessions=True, runs=True, labels=True)
            return 0
        dest = args.destination
        interactive = not args.force
        sids = args.session
        rids = args.run
        context.export(dest, session_ids=sids, run_ids=rids, interactive=interactive)
