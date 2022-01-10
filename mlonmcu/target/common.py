import logging
import subprocess
import argparse

from mlonmcu.cli.helper.parse import extract_features, extract_config

logger = logging.getLogger("mlonmcu")
# logger.setLevel(logging.DEBUG)

def execute(*args, ignore_output=False, live=False, print_func=print, err_func=logger.error, **kwargs):
    logger.info("- Executing: " + str(args))
    if ignore_output:
        assert not live
        subprocess.run([i for i in args], **kwargs, check=True)
        return None
    else:
        outStr = ""
        if live:
            process = subprocess.Popen([i for i in args], **kwargs, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in process.stdout:
                new_line = line.decode(errors="replace")
                outStr = outStr + new_line
                print_func(new_line.replace("\n", ""))
            assert process.poll() == 0, "The process returned an non-zero exit code! (CMD: `{}`)".format(" ".join(args))
        else:
            try:
                p = subprocess.run(
                    [i for i in args], **kwargs, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
                )
                outStr = p.stdout.decode(errors="replace")
                print_func(outStr)
            except subprocess.CalledProcessError as e:
                outStr = e.output.decode(errors="replace")
                err_func(outStr)
                raise

        return outStr


def add_common_options(parser):
    target_group = parser.add_argument_group("target options")
    target_group.add_argument(
         "-f",
         "--feature",
         type=str,
         metavar="FEATURE",
         action="append",
         # choices=list(dict.fromkeys(ALL_FEATURES)), # TODO: get from selected target?
         help="Enabled features for target")
    target_group.add_argument("-c", "--config",
                         metavar="KEY=VALUE",
                         nargs='+',
                         action="append",
                         help="Custom target config as key-value pairs")

def cli(target, args=None):
    print("CLI")
    parser = argparse.ArgumentParser(description=f"ML on MCU Target")
    subparsers = parser.add_subparsers(dest="subcommand")
    exec_parser = subparsers.add_parser("exec", description="Run program with target")
    def _handle_execute(args):
        print("_handle_execute")
        t = target(features=extract_features(args), config=extract_config(args))
        t.exec(args.program, *args.extra_args, live=True)
    exec_parser.set_defaults(func=_handle_execute)
    add_common_options(exec_parser)
    exec_group = exec_parser.add_argument_group("Exec options")
    exec_group.add_argument("program", metavar="EXE", type=str, help="The program which should be executed")
    exec_group.add_argument("extra_args", metavar="ARG", nargs="*", help="Additional arguments")
    inspect_parser = subparsers.add_parser("inspect", description="Inspect program with target")
    def _handle_inspect(args):
        print("_handle_inspect")
        t = target(features=extract_features(args), config=extract_config(args))
        t.inspect(args.program)
    inspect_parser.set_defaults(func=_handle_inspect)
    add_common_options(inspect_parser)
    inspect_group = inspect_parser.add_argument_group("Inspect options")
    inspect_group.add_argument("program", metavar="EXE", type=str, help="The program which should be inspected")
    if args:
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()
    print("args", args)
    if hasattr(args, "func"):
        args.func(args)
    else:
        raise RuntimeError("Invalid command. For usage details use '--help'!")


    # Argparge
    # Env Vars
    # MLONMCU Context (can be disabled via --plain)
