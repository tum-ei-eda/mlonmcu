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

from .platform import CompilePlatform, TargetPlatform

logger = get_logger()


class EspIdfPlatform(CompilePlatform, TargetPlatform):
    """ESP-IDF Platform class."""

    FEATURES = CompilePlatform.FEATURES + TargetPlatform.FEATURES + []

    DEFAULTS = {
        **CompilePlatform.DEFAULTS,
        **TargetPlatform.DEFAULTS,
        "project_template": None,
        "port": None,
        "baud": None,
    }

    # REQUIRED = ["espidf.dir", "espidf.project_template"]
    REQUIRED = []  # For now just expect the user to be already in an esp-idf environment

    def __init__(self, framework, backend, target, features=None, config=None, context=None):
        super().__init__(
            "espidf",
            framework=framework,
            backend=backend,
            target=target,
            features=features,
            config=config,
            context=context,
        )
        self.tempdir = None
        self.project_name = "app"
        dir_name = self.name
        # if self.config["project_dir"]:
        use_provided_dir = False
        if use_provided_dir:
            self.project_dir = Path(self.config["project_dir"])
        else:
            if context:
                assert "temp" in context.environment.paths
                self.project_dir = (
                    context.environment.paths["temp"].path / dir_name
                )  # TODO: Need to lock this for parallel builds
            else:
                logger.info(
                    "Creating temporary directory because no context was available"
                    "and 'espidf.project_dir' was not supplied"
                )
                self.tempdir = tempfile.TemporaryDirectory()
                self.project_dir = Path(self.tempdir.name) / dir_name
                logger.info("Temporary project directory: %s", self.project_dir)
        self.idf_exe = "idf.py"

    def set_directory(self, directory):
        self.project_dir = directory

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
        # try:
        #
        #     subprocess.run(["which", self.idf_exe], shell=True, check=True, stdout=subprocess.PIPE)
        # except subprocess.CalledProcessError as e:
        #     raise RuntimeError(
        #         f"It seems like '{self.idf_exe}' is not available. Make sure to setup your environment!"
        #     ) from e

    # def prepare(self, model, ignore_data=False):
    def prepare(self, src, num=1):
        self.check()
        template_dir = self.project_template
        assert template_dir is not None, "No espidf.project_template was provided"  # TODO: fallback to default one?
        template_dir = Path(template_dir)
        assert template_dir.is_dir(), f"Provided project template does not exists: {template_dir}"
        shutil.copytree(template_dir, self.project_dir)

        def write_defaults(filename):
            defs = {}
            self.framework.add_espidf_defs(defs)
            self.backend.add_espidf_defs(defs)
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
                framework_upper = self.framework.name.upper()
                f.write(f"CONFIG_MLONMCU_FRAMEWORK_{framework_upper}=y\n")
                backend_upper = self.backend.name.upper()
                f.write(f"CONFIG_MLONMCU_BACKEND_{backend_upper}=y\n")
                f.write(f"CONFIG_MLONMCU_NUM_RUNS={num}\n")
                for key, value in defs.items():
                    f.write(f'CONFIG_{key}="{value}"\n')

        write_defaults(self.project_dir / "sdkconfig.defaults")
        idfArgs = [
            self.idf_exe,
            "-C",
            self.project_dir,
            "set-target",
            self.target.name,
        ]
        utils.exec_getout(*idfArgs, live=self.print_output)

    def get_idf_cmake_args(self):
        cmake_defs = {"CMAKE_BUILD_TYPE": "Debug" if self.debug else "Release"}
        return [f"-D{key}={value}" for key, value in cmake_defs.items()]

    def compile(self, src=None, num=1):
        # TODO: build with cmake options
        self.prepare(src, num=num)
        # TODO: support self.num_threads (e.g. patch esp-idf)
        idfArgs = [
            self.idf_exe,
            "-C",
            self.project_dir,
            *self.get_idf_cmake_args(),
            "build",
        ]
        utils.exec_getout(*idfArgs, live=self.print_output)

    def generate_elf(self, src=None, model=None, num=1, data_file=None):
        artifacts = []
        if num > 1:
            raise NotImplementedError
        self.compile(src=src, num=num)
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

    def flash(self, timeout=120):
        # TODO: implement timeout
        # TODO: make sure that already compiled? -> error or just call compile routine?
        input(f"Make sure that the device '{self.target.name}' is connected before you press Enter")
        idfArgs = [
            self.idf_exe,
            "-C",
            self.project_dir,
            *self.get_idf_cmake_args(),
            "flash",
            *self.get_idf_serial_args(),
        ]
        utils.exec_getout(*idfArgs, live=self.print_output)

    def monitor(self, timeout=60):
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
