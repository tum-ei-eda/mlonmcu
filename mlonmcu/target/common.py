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

import logging
import subprocess
import argparse
from typing import List, Callable

from mlonmcu.cli.helper.parse import extract_feature_names, extract_config
from mlonmcu.feature.type import FeatureType
from mlonmcu.feature.features import get_available_feature_names, get_available_features
from mlonmcu.logging import get_logger

logger = get_logger()


# TODO: merge together with mlonmcu.setup.utils.exec_getout
def execute(
    *args: List[str],
    ignore_output: bool = False,
    live: bool = False,
    print_func: Callable = print,
    handle_exit = None,
    err_func: Callable = logger.error,
    **kwargs,
) -> str:
    """Wrapper for running a program in a subprocess.

    Parameters
    ----------
    args : list
        The actual command.
    ignore_output : bool
        Do not get the stdout and stderr or the subprocess.
    live : bool
        Print the output line by line instead of only at the end.
    print_func : Callable
        Function which should be used to print sysout messages.
    err_func : Callable
        Function which should be used to print errors.
    kwargs: dict
        Arbitrary keyword arguments passed through to the subprocess.

    Returns
    -------
    out : str
        The command line output of the command
    """
    logger.debug("- Executing: %s", str(args))
    if ignore_output:
        assert not live
        subprocess.run(args, **kwargs, check=True)
        return None

    out_str = ""
    if live:
        with subprocess.Popen(
            args,
            **kwargs,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ) as process:
            for line in process.stdout:
                new_line = line.decode(errors="replace")
                out_str = out_str + new_line
                print_func(new_line.replace("\n", ""))
            exit_code = None
            while exit_code is None:
                exit_code = process.poll()
            if handle_exit is not None:
                exit_code = handle_exit(exit_code)
            assert exit_code == 0, "The process returned an non-zero exit code {}! (CMD: `{}`)".format(
                exit_code, " ".join(list(map(str, args)))
            )
    else:
        p = subprocess.Popen([i for i in args], **kwargs, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out_str = p.communicate()[0].decode(errors="replace")
        exit_code = p.poll()
        print_func(out_str)
        if handle_exit is not None:
            exit_code = handle_exit(exit_code)
        if exit_code != 0:
            err_func(out_Str)
        assert exit_code == 0, "The process returned an non-zero exit code {}! (CMD: `{}`)".format(
            exit_code, " ".join(list(map(str, args)))
        )

    return out_str


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
        feature_classes = get_available_features(feature_type=FeatureType.TARGET, feature_name=name)
        for feature_class in feature_classes:
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
        config = extract_config(args)
        feature_names = extract_feature_names(args)
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
        config = extract_config(args)
        feature_names = extract_feature_names(args)
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
