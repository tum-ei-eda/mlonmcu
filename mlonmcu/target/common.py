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
"""Helper functions used by MLonMCU targets"""

import argparse
from typing import List

from mlonmcu.cli.helper.parse import extract_feature_names, extract_config
from mlonmcu.feature.type import FeatureType
from mlonmcu.feature.features import get_available_features
from mlonmcu.logging import get_logger
from mlonmcu.setup.utils import execute as execute_new

logger = get_logger()


def execute(*args, **kwargs):
    """Redicrection of old mlonmcu.target.common.execute to new location
    mlonmcu.setup.utils.execute

    Parameters
    ----------
    args : list
        Arguments
    kwargs : dict
        Keyword Arguments
    """
    logger.warning("DEPRECATED: Please use mlonmcu.setup.utils.execute instead of mlonmcu.target.common.execute")
    return execute_new(*args, **kwargs)


def add_common_options(parser: argparse.ArgumentParser, target):
    """Add a set of common options to a command line parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The command line parser
    """
    target_group = parser.add_argument_group("target options")
    target_group.add_argument(
        "-f",
        "--feature",
        type=str,
        metavar="FEATURE",
        action="append",
        choices=target.FEATURES,  # TODO: get from selected target?
        help="Enabled features for target (Choices: %(choices)s)",
    )
    target_group.add_argument(
        "-c",
        "--config",
        metavar="KEY=VALUE",
        nargs="+",
        action="append",
        help="Custom target config as key-value pairs",
    )


def init_target_features(names, config):
    features = []
    for name in names:
        avail_features = get_available_features(feature_type=FeatureType.TARGET, feature_name=name)
        for feature_class in avail_features.values():
            features.append(feature_class(config=config))
    return features


def cli(target, args: List[str] = None):
    """Utility to handle the command line api for targets.

    Parameters
    ----------
    target : Target
        The target to be used.
    args : list
        Interface to pass arguments to the command line parser from test functions.
    """
    parser = argparse.ArgumentParser(description="ML on MCU Target")
    subparsers = parser.add_subparsers(dest="subcommand")
    exec_parser = subparsers.add_parser("exec", description="Run program with target")

    def _handle_execute(args):
        config, _ = extract_config(args)
        feature_names, _ = extract_feature_names(args)
        features = init_target_features(feature_names, config)
        target_inst = target(features=features, config=config)
        target_inst.exec(args.program, *args.extra_args, live=True)

    exec_parser.set_defaults(func=_handle_execute)
    add_common_options(exec_parser, target)
    exec_group = exec_parser.add_argument_group("Exec options")
    exec_group.add_argument("program", metavar="EXE", type=str, help="The program which should be executed")
    exec_group.add_argument("extra_args", metavar="ARG", nargs="*", help="Additional arguments")
    inspect_parser = subparsers.add_parser("inspect", description="Inspect program with target")

    def _handle_inspect(args):
        config, _ = extract_config(args)
        feature_names, _ = extract_feature_names(args)
        features = init_target_features(feature_names, config)
        target_inst = target(features=features, config=config)
        target_inst.inspect(args.program)

    inspect_parser.set_defaults(func=_handle_inspect)
    add_common_options(inspect_parser, target)
    inspect_group = inspect_parser.add_argument_group("Inspect options")
    inspect_group.add_argument("program", metavar="EXE", type=str, help="The program which should be inspected")
    if args:
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        raise RuntimeError("Invalid command. For usage details use '--help'!")

    # Argparge
    # Env Vars
    # MLONMCU Context (can be disabled via --plain)
