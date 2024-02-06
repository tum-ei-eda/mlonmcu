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
import os
import sys
import multiprocessing
import logging
import argparse

from mlonmcu.platform import get_platforms
from mlonmcu.session.postprocess import SUPPORTED_POSTPROCESSES
from mlonmcu.feature.features import get_available_feature_names
from mlonmcu.logging import get_logger, set_log_level
from .helper.parse import extract_config

logger = get_logger()

NUM_GEN_ARGS = 9


def handle_logging_flags(args):
    if hasattr(args, "verbose") and hasattr(args, "quiet"):
        if args.verbose and args.quiet:
            raise RuntimeError("--verbose and --quiet can not be used at the same time")
        elif args.verbose:
            set_log_level(logging.DEBUG)
        elif args.quiet:
            set_log_level(logging.WARNING)
        else:
            set_log_level(logging.INFO)


def add_flow_options(parser):
    flow_parser = parser.add_argument_group("flow options")
    flow_parser.add_argument(  # TODO: move to compile.py?
        "-t",
        "--target",
        type=str,
        metavar="TARGET",
        # choices=SUPPORTED_TARGETS.keys(),
        action="append",
        # default=None,
        # nargs=1,
        help="The target device/architecture (choices: See --list-targets)",
    )
    flow_parser.add_argument(  # TODO: move to compile.py?
        "--list-targets",
        action="store_true",
        help="List the supported targets in the environment",
    )
    flow_parser.add_argument(
        # "-p",
        "--platform",
        type=str,
        metavar="PLATFORM",
        choices=get_platforms().keys(),
        default=None,
        action="append",
        nargs=1,
        help="Explicitly choose the platforms to use (choices: %(choices)s)",
    )
    flow_parser.add_argument(
        # "-p",
        "--postprocess",
        type=str,
        metavar="POSTPROCESS",
        choices=SUPPORTED_POSTPROCESSES.keys(),
        action="append",
        help="Choose the postprocesses to apply (choices: %(choices)s)",
    )
    flow_parser.add_argument(
        "-f",
        "--feature",
        type=str,
        metavar="FEATURE",
        # nargs=1,
        action="append",
        choices=get_available_feature_names(),
        help="Enabled features for target/framework/backend (choices: %(choices)s)",
    )
    flow_parser.add_argument(
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

    def add_gen_args(parser, number):
        for i in range(number):
            suffix = str(i + 1) if i > 0 else ""
            parser.add_argument(
                "--feature-gen" + suffix,
                dest="feature_gen" + suffix,
                type=str,
                metavar="FEATURES",
                nargs="+",
                action="append",
                help=(
                    f"Generator statement for features. (Also available: --feature-gen2 ... --feature-gen{number})"
                    if i == 0
                    else argparse.SUPPRESS
                ),
            )
            flow_parser.add_argument(
                "--config-gen" + suffix,
                dest="config_gen" + suffix,
                metavar="KEY=VALUE",
                nargs="+",
                action="append",
                help=(
                    f"Generator statement for configs. (Also available: --config-gen2 ... --config-gen{number})"
                    if i == 0
                    else argparse.SUPPRESS
                ),
            )

    add_gen_args(flow_parser, NUM_GEN_ARGS)
    flow_parser.add_argument(
        "--parallel",
        metavar="THREADS",
        nargs="?",
        type=int,
        const=multiprocessing.cpu_count(),
        default=1,
        help="Use multiple threads to process runs in parallel (%(const)s if specified, else %(default)s)",
    )
    flow_parser.add_argument(
        "-p",
        "--progress",
        action="store_true",
        help="Display progress bar (default: %(default)s)",
    )
    flow_parser.add_argument(
        "--resume",
        action="store_true",
        help="Try to resume the latest session (default: %(default)s)",
    )
    flow_parser.add_argument(  # TODO: move to compile.py?
        "-l",
        "--label",
        type=str,
        metavar="LABEL",
        default="",
        help="Label for the session (default: %(default)s)",
    )


def add_common_options(parser):
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed messages for easier debugging (default: %(default)s)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Reduce number of logging statements to a minimum (default: %(default)s)",
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Execute run.py inside docker container (default: %(default)s)",
    )


def add_context_options(parser, with_home=True):
    common = parser.add_argument_group("context options")
    if with_home:
        home = os.getenv("MLONMCU_HOME", None)
        common.add_argument(
            "-H",
            "--home",
            "--hint",
            type=str,
            default=home,
            help="The path to the mlonmcu environment (overwriting $MLONMCU_HOME environment variable)",
        )


def add_model_options(parser):
    parser.add_argument(
        "models",
        metavar="model",
        type=str,
        nargs="+",
        default=None,
        help="Model to process",
    )


def kickoff_runs(args, until, context):
    assert len(context.sessions) > 0
    session = context.sessions[-1]
    # session.label = args.label
    config = extract_config(args)
    # TODO: move into context/session
    per_stage = True
    print_report = True
    if "runs_per_stage" in config:
        per_stage = bool(config["runs_per_stage"])
    elif "runs_per_stage" in context.environment.vars:
        per_stage = bool(context.environment.vars["runs_per_stage"])
    if "print_report" in config:
        print_report = bool(config["print_report"])
    elif "print_report" in context.environment.vars:
        print_report = bool(context.environment.vars["print_report"])
    with session:
        success = session.process_runs(
            until=until,
            per_stage=per_stage,
            print_report=print_report,
            num_workers=args.parallel,
            progress=args.progress,
            context=context,
            export=True,
        )
    if not success:
        logger.error("At least one error occured!")
        sys.exit(1)
