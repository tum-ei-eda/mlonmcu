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
"""Console script for mlonmcu."""

import os
import argparse
import sys
import subprocess
import platform


from mlonmcu.logging import get_logger
import mlonmcu.cli.init as init
import mlonmcu.cli.setup as setup
import mlonmcu.cli.flow as flow
import mlonmcu.cli.cleanup as cleanup
import mlonmcu.cli.export as export
import mlonmcu.cli.env as env
import mlonmcu.cli.models as models
from .common import handle_logging_flags, add_common_options
from ..version import __version__

logger = get_logger()


def handle_docker(args):
    if args.docker:
        home = os.environ.get("MLONMCU_HOME")
        assert home is not None, (
            "To use the --docker functionality, please export the MLONMCU_HOME environment variable"
            + " to a directory which should be mounted by the container"
        )
        exec_args = sys.argv[1:]
        exec_args.remove("--docker")
        docker = subprocess.Popen(
            [
                "docker-compose",
                "-f",
                "docker/docker-compose.yml",
                "run",
                "-e",
                f"MLONMCU_HOME={home}",
                "--rm",
                "mlonmcu",
                "python3",
                "-m",
                "mlonmcu.cli.main",
                *exec_args,
            ],
            env={"MLONMCU_HOME": home},
        )
        stdout, stderr = docker.communicate()
        exit_code = docker.wait()
        if exit_code > 0:
            logger.warning(f"Docker compose process completed with exit code: {exit_code}")
        sys.exit(exit_code)

    if platform.system() in ["Darwin", "Windows"]:
        raise RuntimeError(
            "Only Linux is supported at the Moment. If you have Docker installed, you may want to"
            + " try running this script using the `--docker` flag."
        )


# def main(args):
def main(args=None):
    """Console script for mlonmcu."""

    parser = argparse.ArgumentParser(
        description="ML on MCU Flow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # parser.add_argument('_', nargs='*')
    parser.add_argument("-V", "--version", action="version", version="mlonmcu " + __version__)
    add_common_options(parser)
    subparsers = parser.add_subparsers(dest="subcommand")  # this line changed
    init.get_parser(subparsers)
    setup.get_parser(subparsers)
    flow.get_parser(subparsers)
    cleanup.get_parser(subparsers)
    export.get_parser(subparsers)
    env.get_parser(subparsers)
    models.get_parser(subparsers)
    if args:
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()
    handle_logging_flags(args)
    handle_docker(args)

    if hasattr(args, "func"):
        args.func(args)
    else:
        print("Invalid subcommand for `mlonmcu`!")
        parser.print_help(sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main(args=sys.argv[1:]))  # pragma: no cover
