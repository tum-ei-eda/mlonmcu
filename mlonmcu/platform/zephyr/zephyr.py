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
"""Zephyr Platform"""

import re
import os
import time
import shutil
import tempfile
from pathlib import Path
import pkg_resources
from typing import Tuple


from mlonmcu.setup import utils
from mlonmcu.artifact import Artifact, ArtifactFormat
from mlonmcu.logging import get_logger
from mlonmcu.target import get_targets
from mlonmcu.target.target import Target
from mlonmcu.config import str2bool

from ..platform import CompilePlatform, TargetPlatform
from .zephyr_target import create_zephyr_platform_target

logger = get_logger()


def get_project_template(name="project2"):  # Workaround which only support tvmaot!!!
    zephyr_templates = pkg_resources.resource_listdir("mlonmcu", os.path.join("..", "resources", "platforms", "zephyr"))
    if name not in zephyr_templates:
        return None
    fname = pkg_resources.resource_filename("mlonmcu", os.path.join("..", "resources", "platforms", "zephyr", name))
    return fname


class ZephyrPlatform(CompilePlatform, TargetPlatform):
    """Zephyr Platform class."""

    FEATURES = CompilePlatform.FEATURES | TargetPlatform.FEATURES | {"benchmark"}

    DEFAULTS = {
        **CompilePlatform.DEFAULTS,
        **TargetPlatform.DEFAULTS,
        "project_template": None,
        "project_dir": None,
        "port": None,
        # "port": "/dev/ttyUSB0",
        "baud": 115200,
        "wait_for_user": True,
        "flash_only": False,
        "optimize": None,  # values: 0,1,2,3,s
    }

    REQUIRED = {"zephyr.install_dir", "zephyr.sdk_dir", "zephyr.venv_dir"}

    def __init__(self, features=None, config=None):
        super().__init__(
            "zephyr",
            features=features,
            config=config,
        )
        self.tempdir = None
        self.project_name = "app"
        self.project_dir = None

    @property
    def zephyr_install_dir(self):
        return Path(self.config["zephyr.install_dir"])

    @property
    def zephyr_sdk_dir(self):
        return Path(self.config["zephyr.sdk_dir"])

    @property
    def zephyr_venv_dir(self):
        return Path(self.config["zephyr.venv_dir"])

    @property
    def wait_for_user(self):
        value = self.config["wait_for_user"]
        return str2bool(value)

    @property
    def flash_only(self):
        # TODO: get rid of this
        value = self.config["flash_only"]
        return str2bool(value)

    @property
    def optimize(self):
        return self.config["optimize"]

    def invoke_west(self, *args, **kwargs):
        env = os.environ.copy()
        env["ZEPHYR_BASE"] = str(self.zephyr_install_dir / "zephyr")
        env["ZEPHYR_SDK_INSTALL_DIR"] = str(self.zephyr_sdk_dir)
        cmd = ". " + str(self.zephyr_venv_dir / "bin" / "activate") + " && west " + " ".join([str(arg) for arg in args])
        out = utils.execute(
            cmd, shell=True, env=env, **kwargs, executable="/bin/bash"
        )  # TODO: using shell=True is insecure but right now we can not avoid it?
        return out

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
                    "and 'zephyr.project_dir' was not supplied"
                )
                self.tempdir = tempfile.TemporaryDirectory()
                self.project_dir = Path(self.tempdir.name) / dir_name
                logger.debug("Temporary project directory: %s", self.project_dir)
        self.project_dir.mkdir(exist_ok=True)

    def get_supported_targets(self):
        with tempfile.TemporaryDirectory() as temp:
            f = Path(temp) / "CMakeLists.txt"
            # f.touch()
            with open(f, "w") as handle:
                handle.write(
                    """
cmake_minimum_required(VERSION 3.22)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(ProjectName)
                """
                )
            b = Path(temp) / "build"

            # This will fail
            # Workaround used to get list of supported boards...
            def _handle(code):
                return 0

            text = self.invoke_west("build", "-d", b, "-b", "help", temp, live=False, handle_exit=_handle)
        # Warning: This will fail if a python executable is NOT available in the system. Aliasing
        # python3 to python will not work. Not sure how this would handle a system which only has python2 installed?
        target_names = re.compile(r"^  (\S+)$", re.MULTILINE).findall(text)

        return [
            f"zephyr_{name}" for name in target_names if len(name) > 0 and " " not in name
        ]  # TODO: consider dropping zephyr_ prefix

    def create_target(self, name):
        assert name in self.get_supported_targets(), f"{name} is not a valid Zephyr target"
        targets = get_targets()
        if name in targets:
            base = targets[name]
        else:
            base = Target
        return create_zephyr_platform_target(name, self, base=base)

    @property
    def project_template(self):
        return self.config["project_template"]

    @property
    def port(self):
        return self.config["port"]

    @property
    def baud(self):
        return self.config["baud"]

    @property
    def build_dir(self):
        return self.project_dir / "build"

    def close(self):
        if self.tempdir:
            self.tempdir.cleanup()

    def get_west_cmake_args(self):
        cmake_defs = {"CMAKE_BUILD_TYPE": "Debug" if self.debug else "Release"}
        return [f"-D{key}={value}" for key, value in cmake_defs.items()]

    def prepare(self, target, src):
        self.init_directory()
        template_dir = self.project_template
        if template_dir is None:
            template_dir = get_project_template()
        else:
            template_dir = Path(template_dir)
            if not template_dir.is_dir():
                template_dir = get_project_template(name=str(template_dir))
        assert template_dir is not None, f"Provided project template does not exists: {template_dir}"
        shutil.copytree(template_dir, self.project_dir, dirs_exist_ok=True)

        def write_defaults(filename):
            defs = self.definitions
            with open(filename, "w", encoding="utf-8") as f:
                f.write("CONFIG_CPLUSPLUS=y\n")
                f.write("CONFIG_NEWLIB_LIBC=y\n")
                f.write("CONFIG_REBOOT=y\n")
                f.write("CONFIG_STD_CPP20=y\n")
                f.write("CONFIG_LIB_CPLUSPLUS=y\n")
                f.write("CONFIG_TIMING_FUNCTIONS=y\n")
                # f.write("CONFIG_NEWLIB_LIBC_FLOAT_PRINTF=y\n")
                # f.write("CONFIG_CBPRINTF_FP_SUPPORT=y\n")
                if self.debug:
                    f.write("CONFIG_DEBUG=y\n")
                    if not self.optimize:
                        f.write("CONFIG_DEBUG_OPTIMIZATIONS=y\n")
                else:
                    f.write("CONFIG_DEBUG=n\n")
                    if self.optimize:
                        if str(self.optimize) == "s":
                            f.write("CONFIG_SIZE_OPTIMIZATIONS=y\n")
                        elif str(self.optimize) == "0":
                            f.write("CONFIG_NO_OPTIMIZATIONS=y\n")
                        elif str(self.optimize) == "g":
                            f.write("CONFIG_DEBUG_OPTIMIZATIONS=y\n")
                        elif str(self.optimize) == "2":
                            f.write("CONFIG_SPEED_OPTIMIZATIONS=y\n")
                        elif str(self.optimize) == "3":
                            f.write("CONFIG_SPEED_OPTIMIZATIONS=y\n")
                            f.write('CONFIG_COMPILER_OPT="-O3"\n')
                        else:
                            raise RuntimeError(f"Unsupported optimization level for Zephyr platform: {self.optimize}")
                # f.write("CONFIG_PARTITION_TABLE_SINGLE_APP_LARGE=y\n")
                # if self.debug:
                #     f.write("CONFIG_OPTIMIZATION_LEVEL_DEBUG=y\n")
                #     f.write("CONFIG_COMPILER_OPTIMIZATION_LEVEL_DEBUG=y\n")
                # else:
                #     # Trying to reduce the binary size as much as possible
                #     # (https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-guides/performance/size.html)
                #     f.write("CONFIG_BOOTLOADER_LOG_LEVEL_NONE=y\n")
                #     f.write("CONFIG_BOOTLOADER_LOG_LEVEL=0\n")
                #     f.write("CONFIG_BOOT_ROM_LOG_ALWAYS_OFF=y\n")
                #     f.write("CONFIG_COMPILER_OPTIMIZATION_ASSERTIONS_DISABLE=y\n")
                #     f.write("CONFIG_COMPILER_OPTIMIZATION_ASSERTION_LEVEL=0\n")
                #     f.write("CONFIG_COMPILER_OPTIMIZATION_CHECKS_SILENT=y\n")
                #     f.write("CONFIG_HAL_DEFAULT_ASSERTION_LEVEL=0\n")
                #     f.write("CONFIG_LOG_DEFAULT_LEVEL_NONE=y\n")
                #     f.write("CONFIG_LOG_DEFAULT_LEVEL=0\n")
                #     f.write("CONFIG_LOG_MAXIMUM_LEVEL=0\n")
                #     f.write("CONFIG_NEWLIB_NANO_FORMAT=y\n")
                #     f.write("CONFIG_COMPILER_OPTIMIZATION_LEVEL_RELEASE=y\n")
                #     f.write("CONFIG_OPTIMIZATION_LEVEL_RELEASE=y\n")
                #     optimize_for_size = True
                #     if optimize_for_size:
                #         f.write("CONFIG_COMPILER_OPTIMIZATION_SIZE=y\n")
                #     else:
                #         f.write("CONFIG_COMPILER_OPTIMIZATION_PERF=y\n")
                # watchdog_sec = 60
                # f.write(f"CONFIG_ESP_TASK_WDT_TIMEOUT_S={watchdog_sec}\n")
                f.write(f'CONFIG_MLONMCU_CODEGEN_DIR="{src}"\n')
                for key, value in defs.items():
                    if isinstance(value, bool):
                        value = "y" if value else "n"
                    else:
                        value = f'"{value}"'
                    f.write(f"CONFIG_{key}={value}\n")

        write_defaults(self.project_dir / "prj.conf")
        zephyr_target = target.name.split("_", 1)[-1]
        westArgs = [  # cmake only
            "build",
            "-d",
            self.build_dir,
            "-b",
            zephyr_target,
            self.project_dir,
            *self.get_west_cmake_args(),
            "-c",
        ]
        return self.invoke_west(*westArgs, live=self.print_outputs)

    def compile(self, target, src=None):
        out = self.prepare(target, src)
        # TODO: build with cmake options
        # TODO: support self.num_threads
        # self.build_dir.mkdir()
        zephyr_target = target.name.split("_", 1)[-1]
        westArgs = [
            "build",
            "-d",
            self.build_dir,
            "-b",
            zephyr_target,
            self.project_dir,
            f"-o=-j{self.num_threads}",
        ]
        out += self.invoke_west(*westArgs, live=self.print_outputs)
        return out

    def generate(self, src, target, model=None) -> Tuple[dict, dict]:
        artifacts = []
        out = self.compile(target, src=src)
        elf_name = "zephyr.elf"
        elf_file = self.project_dir / "build" / "zephyr" / elf_name
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
        metrics = self.get_metrics(elf_file)
        stdout_artifact = Artifact(
            "zephyr_out.log", content=out, fmt=ArtifactFormat.TEXT  # TODO: split into one file per command
        )  # TODO: rename to tvmaot_out.log?
        artifacts.append(stdout_artifact)
        return {"default": artifacts}, {"default": metrics}

    def get_serial(self, target):
        port = target.port
        if port is None:
            # Falling back to 'espidf.baud'
            assert self.port is not None, f"Please provide '{target.name}.port'"
            port = self.port
        baud = target.baud
        if baud is None:
            # Falling back to 'espidf.baud'
            assert self.baud is not None, f"Please provide '{target.name}.baud'"
            baud = self.baud
        return port, baud

    def flash(self, elf, target, timeout=120):
        # Ignore elf, as we use self.project_dir instead
        # TODO: add alternative approach which allows passing elf instead
        if elf is not None:
            logger.debug("Ignoring ELF file for zephyr platform")
        # TODO: implement timeout
        # TODO: make sure that already compiled? -> error or just call compile routine?
        if self.wait_for_user:  # INTERACTIVE
            answer = input(
                f"Make sure that the device '{target.name}' is connected before you press [Enter]"
                + " (Type 'Abort' to cancel)"
            )
            if answer.lower() == "abort":
                return ""
        logger.debug("Flashing target software")

        westArgs = [
            "flash",
            "-d",
            self.build_dir,
        ]
        if "esp32" in target.name:  # TODO: implement this for other runners as well
            port, baud = self.get_serial(target)
            if port:
                westArgs.extend(["--esp-device", port])
            if baud:
                westArgs.extend(["--esp-baud-rate", baud])
        self.invoke_west(*westArgs, live=self.print_outputs)

    def monitor(self, target, timeout=60):
        if self.flash_only:
            return ""

        port, baud = self.get_serial(target)

        def _monitor_helper(port, baud, verbose=False, start_match=None, end_match=None, timeout=60):
            # Local import to get rid of pyserial dependency

            import serial

            # start_match and end_match are inclusive
            found_start = start_match is None
            outStr = ""
            if timeout:
                pass  # TODO: implement timeout
            # The following is a custom initialization sequence inspired by
            # (https://github.com/espressif/esp-idf/blob/master/tools/idf_monitor_base/serial_reader.py)
            # Required for esp32c3!
            high = False
            low = True
            ser = serial.Serial(port, baud)
            ser.close()
            ser.dtr = False
            ser.rts = False
            time.sleep(1)
            ser.dtr = low
            ser.rts = high
            ser.dtr = ser.dtr
            ser.open()
            ser.dtr = high
            ser.rts = low
            ser.dtr = ser.dtr
            time.sleep(0.002)
            ser.rts = high
            ser.dtr = ser.dtr
            try:
                while True:
                    try:
                        ser_bytes = ser.readline()
                        new_line = ser_bytes.decode("utf-8", errors="replace")
                        if verbose:
                            print(new_line.replace("\n", ""))
                        if start_match and start_match in new_line:
                            outStr = new_line
                            found_start = True
                        else:
                            outStr = outStr + new_line
                        if found_start:
                            if end_match and end_match in new_line:
                                break
                    except KeyboardInterrupt:
                        logger.warning("Stopped processing serial port (KeyboardInterrupt)")
            finally:
                ser.close()
            return outStr

        logger.debug("Monitoring target software")
        # TODO: implement timeout
        return _monitor_helper(
            port,
            baud,
            verbose=self.print_outputs,
            start_match="MLonMCU: START",
            end_match="MLonMCU: STOP",
            timeout=timeout,
        )
