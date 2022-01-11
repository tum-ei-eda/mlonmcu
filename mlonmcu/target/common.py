"""Helper functions used by MLonMCU targets"""

import logging
import subprocess
import argparse
from typing import List, Callable

from mlonmcu.cli.helper.parse import extract_features, extract_config

logger = logging.getLogger("mlonmcu")


def execute(
    *args: List[str],
    ignore_output: bool = False,
    live: bool = False,
    print_func: Callable = print,
    err_func: Callable = logger.error,
    **kwargs,
) -> str:
    """Wrapper for running a program in a subprocess.
    Parameters
    ----------
    args
        The actual command.
    ignore_output : bool
        Do not get the stdout and stderr or the subprocess.
    live : bool
        Print the output line by line instead of only at the end.
    print_func : Callable
        Function which should be used to print sysout messages.
    err_func : Callable
        Function which should be used to print errors.
    kwargs:
        Arbitrary keyword arguments passed through to the subprocess.
    Returns
    -------
    out : str
        The command line output of the command
    """
    logger.info("- Executing: %s", str(args))
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
            assert process.poll() == 0, (
                "The process returned an non-zero exit code! (CMD: `"
                + " ".join(args)
                + "`)"
            )
    else:
        try:
            process = subprocess.run(
                args,
                **kwargs,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            out_str = process.stdout.decode(errors="replace")
            print_func(out_str)
        except subprocess.CalledProcessError as err:
            out_str = err.output.decode(errors="replace")
            err_func(out_str)
            raise

    return out_str


def add_common_options(parser: argparse.ArgumentParser):
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
        # choices=list(dict.fromkeys(ALL_FEATURES)), # TODO: get from selected target?
        help="Enabled features for target",
    )
    target_group.add_argument(
        "-c",
        "--config",
        metavar="KEY=VALUE",
        nargs="+",
        action="append",
        help="Custom target config as key-value pairs",
    )


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
        target_inst = target(
            features=extract_features(args), config=extract_config(args)
        )
        target_inst.exec(args.program, *args.extra_args, live=True)

    exec_parser.set_defaults(func=_handle_execute)
    add_common_options(exec_parser)
    exec_group = exec_parser.add_argument_group("Exec options")
    exec_group.add_argument(
        "program", metavar="EXE", type=str, help="The program which should be executed"
    )
    exec_group.add_argument(
        "extra_args", metavar="ARG", nargs="*", help="Additional arguments"
    )
    inspect_parser = subparsers.add_parser(
        "inspect", description="Inspect program with target"
    )

    def _handle_inspect(args):
        target_inst = target(
            features=extract_features(args), config=extract_config(args)
        )
        target_inst.inspect(args.program)

    inspect_parser.set_defaults(func=_handle_inspect)
    add_common_options(inspect_parser)
    inspect_group = inspect_parser.add_argument_group("Inspect options")
    inspect_group.add_argument(
        "program", metavar="EXE", type=str, help="The program which should be inspected"
    )
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
