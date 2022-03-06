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
"""ESP-IDF Platform"""

import os
import signal
import shutil
import tempfile
import subprocess
from pathlib import Path

import psutil

from mlonmcu.setup import utils
from mlonmcu.artifact import Artifact, ArtifactFormat
from mlonmcu.logging import get_logger
from mlonmcu.target import SUPPORTED_TARGETS
from mlonmcu.target.target import Target

from .platform import CompilePlatform, TargetPlatform
from .espidf_target import create_espidf_target

logger = get_logger()

import pkgutil
import os
import pkg_resources

def get_project_template(name="project"):
    espidf_templates = pkg_resources.resource_listdir("mlonmcu", os.path.join("..", "resources", "platforms", "espidf"))
    if name not in espidf_templates:
        return None
    fname = pkg_resources.resource_filename("mlonmcu", os.path.join("..", "resources", "platforms", "espidf", name))
    return fname


class EspIdfPlatform(CompilePlatform, TargetPlatform):
    """ESP-IDF Platform class."""

    FEATURES = CompilePlatform.FEATURES + TargetPlatform.FEATURES + []

    DEFAULTS = {
        **CompilePlatform.DEFAULTS,
        **TargetPlatform.DEFAULTS,
        "project_template": None,
        "project_dir": None,
        "port": None,
        "baud": None,
    }

    # REQUIRED = ["espidf.dir", "espidf.project_template"]
    REQUIRED = []  # For now just expect the user to be already in an esp-idf environment

    # def __init__(self, framework, backend, target, features=None, config=None, context=None):
    def __init__(self, features=None, config=None):
        super().__init__(
            "espidf",
            # Framework=framework,
            # Backend=backend,
            # Target=target,
            features=features,
            config=config,
            # context=context,
        )
        self.tempdir = None
        self.project_name = "app"
        self.project_dir = None
        self.idf_exe = "idf.py"

    def init_directory(self, path=None, context=None):
        if self.project_dir is not None:
            self.project_dir.mkdir(exist_ok=True)
            logger.debug("Project directory already initialized")
            return
        dir_name = self.name
        if path is not None:
            self.project_dir = Path(path)
        elif self.config["project_dir"] is not None:
            self.project_dir = Path(self.config["project_dir"])
        else:
            if context:
                assert "temp" in context.environment.paths
                self.project_dir = (
                    context.environment.paths["temp"].path / dir_name
                )  # TODO: Need to lock this for parallel builds
            else:
                logger.debug(
                    "Creating temporary directory because no context was available "
                    "and 'espidf.project_dir' was not supplied"
                )
                self.tempdir = tempfile.TemporaryDirectory()
                self.project_dir = Path(self.tempdir.name) / dir_name
                logger.debug("Temporary project directory: %s", self.project_dir)
        self.project_dir.mkdir(exist_ok=True)

    def get_supported_targets(self):
        idfArgs = [self.idf_exe, "--list-targets"]
        text = utils.exec_getout(*idfArgs, live=self.print_output, print_output=False)
        target_names = text.split("\n")

        return [name for name in target_names if len(name) > 0]

    def create_target(self, name):
        assert name in self.get_supported_targets(), f"{name} is not a valid ESP-IDF target"
        if name in SUPPORTED_TARGETS:
            base = SUPPORTED_TARGETS[name]
        else:
            base = Target
        return create_espidf_target(name, self, base=base)

    @property
    def project_template(self):
        return self.config["project_template"]

    @property
    def port(self):
        return self.config["port"]

    @property
    def baud(self):
        return self.config["baud"]

    def close(self):
        if self.tempdir:
            self.tempdir.cleanup()

    def check(self):
        if not shutil.which(self.idf_exe):
            raise RuntimeError(f"It seems like '{self.idf_exe}' is not available. Make sure to setup your environment!")

    # def prepare(self, model, ignore_data=False):
    def prepare(self, target, src, num=1):
        self.init_directory()
        self.check()
        template_dir = self.project_template
        if template_dir is None:
            template_dir = get_project_template()
        else:
            template_dir = Path(template_dir)
            if not template_dir.is_dir():
                template_dir = get_project_template(name=template_dir)
        assert template_dir is not None, f"Provided project template does not exists: {template_dir}"
        shutil.copytree(template_dir, self.project_dir, dirs_exist_ok=True)

        def write_defaults(filename):
            defs = self.definitions
            with open(filename, "w", encoding="utf-8") as f:
                f.write("CONFIG_PARTITION_TABLE_SINGLE_APP_LARGE=y\n")
                if self.debug:
                    f.write("CONFIG_OPTIMIZATION_LEVEL_DEBUG=y\n")
                    f.write("CONFIG_COMPILER_OPTIMIZATION_LEVEL_DEBUG=y\n")
                else:
                    f.write("CONFIG_COMPILER_OPTIMIZATION_LEVEL_RELEASE=y\n")
                    f.write("CONFIG_OPTIMIZATION_LEVEL_RELEASE=y\n")
                    optimize_for_size = True
                    if optimize_for_size:
                        f.write("CONFIG_COMPILER_OPTIMIZATION_SIZE=y\n")
                    else:
                        f.write("CONFIG_COMPILER_OPTIMIZATION_PERF=y\n")
                watchdog_sec = 60
                f.write(f"CONFIG_ESP_TASK_WDT_TIMEOUT_S={watchdog_sec}\n")
                f.write(f'CONFIG_MLONMCU_CODEGEN_DIR="{src}"\n')
                f.write(f"CONFIG_MLONMCU_NUM_RUNS={num}\n")
                for key, value in defs.items():
                    if isinstance(value, bool):
                        value = 'y' if value else 'n'
                    else:
                        value = f'"{value}"'
                    f.write(f'CONFIG_{key}={value}\n')

        write_defaults(self.project_dir / "sdkconfig.defaults")
        idfArgs = [
            self.idf_exe,
            "-C",
            self.project_dir,
            "set-target",
            target.name,
        ]
        utils.exec_getout(*idfArgs, live=self.print_output)

    def get_idf_cmake_args(self):
        cmake_defs = {"CMAKE_BUILD_TYPE": "Debug" if self.debug else "Release"}
        return [f"-D{key}={value}" for key, value in cmake_defs.items()]

    def compile(self, target, src=None, num=1):
        # TODO: build with cmake options
        self.prepare(target, src, num=num)
        # TODO: support self.num_threads (e.g. patch esp-idf)
        idfArgs = [
            self.idf_exe,
            "-C",
            self.project_dir,
            *self.get_idf_cmake_args(),
            "build",
        ]
        utils.exec_getout(*idfArgs, live=self.print_output)

    def generate_elf(self, target, src=None, model=None, num=1, data_file=None):
        artifacts = []
        if num > 1:
            raise NotImplementedError
        self.compile(target, src=src, num=num)
        elf_name = self.project_name + ".elf"
        elf_file = self.project_dir / "build" / elf_name
        # TODO: just use path instead of raw data?
        if self.tempdir:
            # Warning: The porject context will get destroyed afterwards wehen using  a temporory directory
            with open(elf_file, "rb") as handle:
                data = handle.read()
                artifact = Artifact("generic_mlif", raw=data, fmt=ArtifactFormat.RAW)
                artifacts.append(artifact)
        else:
            artifact = Artifact(elf_name, path=elf_file, fmt=ArtifactFormat.PATH)
            artifacts.append(artifact)
        self.artifacts = artifacts

    def get_idf_serial_args(self):
        args = []
        if self.port:
            args.extend(["-p", self.port])
        if self.baud:
            args.extend(["-b", self.baud])
        return args

    def flash(self, target, timeout=120):
        # TODO: implement timeout
        # TODO: make sure that already compiled? -> error or just call compile routine?
        input(f"Make sure that the device '{target.name}' is connected before you press Enter")
        idfArgs = [
            self.idf_exe,
            "-C",
            self.project_dir,
            *self.get_idf_cmake_args(),
            "flash",
            *self.get_idf_serial_args(),
        ]
        utils.exec_getout(*idfArgs, live=self.print_output)

    def monitor(self, target, timeout=60):
        def _kill_monitor():

            for proc in psutil.process_iter():
                # check whether the process name matches
                cmdline = " ".join(proc.cmdline())
                if "idf_monitor.py" in cmdline:  # TODO: do something less "dangerous"?
                    proc.kill()

        def _monitor_helper(*args, verbose=False, start_match=None, end_match=None, timeout=60):
            # start_match and end_match are inclusive
            if timeout:
                pass  # TODO: implement timeout
            found_start = start_match is None
            logger.debug("- Executing: %s", str(args))
            outStr = ""
            process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            try:
                exit_code = None
                for line in process.stdout:
                    new_line = line.decode(errors="replace")
                    if verbose:
                        print(new_line.replace("\n", ""))
                    if start_match and start_match in new_line:
                        outStr = new_line
                        found_start = True
                    else:
                        outStr = outStr + new_line
                    if found_start:
                        if end_match and end_match in new_line:
                            _kill_monitor()
                            process.terminate()
                            exit_code = 0
                while exit_code is None:
                    exit_code = process.poll()
                if not verbose and exit_code != 0:
                    logger.error(outStr)
                assert exit_code == 0, "The process returned an non-zero exit code {}! (CMD: `{}`)".format(
                    exit_code, " ".join(list(map(str, args)))
                )
            except KeyboardInterrupt:
                logger.debug("Interrupted subprocess. Sending SIGINT signal...")
                _kill_monitor()
                pid = process.pid
                os.kill(pid, signal.SIGINT)
            return outStr

        # TODO: implement timeout
        idfArgs = [
            self.idf_exe,
            "-C",
            self.project_dir,
            *self.get_idf_cmake_args(),
            "monitor",
            *self.get_idf_serial_args(),
        ]
        return _monitor_helper(
            *idfArgs,
            verbose=self.print_output,
            start_match="MLonMCU: START",
            end_match="MLonMCU: STOP",
            timeout=timeout,
        )
