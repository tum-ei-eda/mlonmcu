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
import time
import signal
import shutil
import tempfile
import subprocess
from pathlib import Path
import pkg_resources


from mlonmcu.setup import utils
from mlonmcu.artifact import Artifact, ArtifactFormat
from mlonmcu.logging import get_logger
from mlonmcu.target.target import Target
from mlonmcu.config import str2bool

from ..platform import CompilePlatform, TargetPlatform
from .espidf_target import create_espidf_platform_target, get_espidf_platform_targets

logger = get_logger()


def get_project_template(name="project"):
    espidf_templates = pkg_resources.resource_listdir("mlonmcu", os.path.join("..", "resources", "platforms", "espidf"))
    if name not in espidf_templates:
        return None
    fname = pkg_resources.resource_filename("mlonmcu", os.path.join("..", "resources", "platforms", "espidf", name))
    return fname


class EspIdfPlatform(CompilePlatform, TargetPlatform):
    """ESP-IDF Platform class."""

    FEATURES = CompilePlatform.FEATURES | TargetPlatform.FEATURES | {"benchmark"}

    DEFAULTS = {
        **CompilePlatform.DEFAULTS,
        **TargetPlatform.DEFAULTS,
        "project_template": None,
        "project_dir": None,
        "port": None,
        "baud": 115200,
        "use_idf_monitor": True,
        "wait_for_user": True,
        "flash_only": False,
    }

    REQUIRED = {"espidf.install_dir", "espidf.src_dir"}

    def __init__(self, features=None, config=None):
        super().__init__(
            "espidf",
            features=features,
            config=config,
        )
        self.tempdir = None
        self.project_name = "app"
        self.project_dir = None

    @property
    def espidf_install_dir(self):
        return Path(self.config["espidf.install_dir"])

    @property
    def espidf_src_dir(self):
        return Path(self.config["espidf.src_dir"])

    @property
    def idf_exe(self):
        return self.espidf_src_dir / "tools" / "idf.py"

    @property
    def use_idf_monitor(self):
        value = self.config["use_idf_monitor"]
        return str2bool(value)

    @property
    def wait_for_user(self):
        value = self.config["wait_for_user"]
        return str2bool(value)

    @property
    def flash_only(self):
        # TODO: get rid of this
        value = self.config["flash_only"]
        return str2bool(value)

    def invoke_idf_exe(self, *args, **kwargs):
        env = os.environ.copy()
        env["IDF_PATH"] = str(self.espidf_src_dir)
        env["IDF_TOOLS_PATH"] = str(self.espidf_install_dir)
        cmd = (
            ". "
            + str(self.espidf_src_dir / "export.sh")
            + f" && {self.idf_exe} "
            # + f" > /dev/null && {self.idf_exe} "
            + " ".join([str(arg) for arg in args])
        )
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
                    "and 'espidf.project_dir' was not supplied"
                )
                self.tempdir = tempfile.TemporaryDirectory()
                self.project_dir = Path(self.tempdir.name) / dir_name
                logger.debug("Temporary project directory: %s", self.project_dir)
        self.project_dir.mkdir(exist_ok=True)

    def get_supported_targets(self):
        text = self.invoke_idf_exe("--list-targets", live=self.print_outputs)
        # Warning: This will fail if a python executable is NOT available in the system. Aliasing
        # python3 to python will not work. Not sure how this would handle a system which only has python2 installed?
        target_names = text.split("\n")

        def filter_names(x):
            ret = []
            for x in reversed(x):
                if len(x) > 0:
                    ret.append(x)
                else:
                    if len(ret) > 0:
                        break
                    else:
                        continue
            return reversed(ret)

        target_names = filter_names(target_names)

        return [name for name in target_names if len(name) > 0 and " " not in name]

    def create_target(self, name):
        assert name in self.get_supported_targets(), f"{name} is not a valid ESP-IDF target"
        targets = get_espidf_platform_targets()
        if name in targets:
            base = targets[name]
        else:
            base = Target
        return create_espidf_platform_target(name, self, base=base)

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

    def prepare(self, target, src):
        self.init_directory()
        self.check()
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
                f.write("CONFIG_PARTITION_TABLE_SINGLE_APP_LARGE=y\n")
                if self.debug:
                    f.write("CONFIG_OPTIMIZATION_LEVEL_DEBUG=y\n")
                    f.write("CONFIG_COMPILER_OPTIMIZATION_LEVEL_DEBUG=y\n")
                else:
                    # Trying to reduce the binary size as much as possible
                    # (https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-guides/performance/size.html)
                    f.write("CONFIG_BOOTLOADER_LOG_LEVEL_NONE=y\n")
                    f.write("CONFIG_BOOTLOADER_LOG_LEVEL=0\n")
                    f.write("CONFIG_BOOT_ROM_LOG_ALWAYS_OFF=y\n")
                    f.write("CONFIG_COMPILER_OPTIMIZATION_ASSERTIONS_DISABLE=y\n")
                    f.write("CONFIG_COMPILER_OPTIMIZATION_ASSERTION_LEVEL=0\n")
                    f.write("CONFIG_COMPILER_OPTIMIZATION_CHECKS_SILENT=y\n")
                    f.write("CONFIG_HAL_DEFAULT_ASSERTION_LEVEL=0\n")
                    f.write("CONFIG_LOG_DEFAULT_LEVEL_NONE=y\n")
                    f.write("CONFIG_LOG_DEFAULT_LEVEL=0\n")
                    f.write("CONFIG_LOG_MAXIMUM_LEVEL=0\n")
                    f.write("CONFIG_NEWLIB_NANO_FORMAT=y\n")
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
                for key, value in defs.items():
                    if isinstance(value, bool):
                        value = "y" if value else "n"
                    else:
                        value = f'"{value}"'
                    f.write(f"CONFIG_{key}={value}\n")

        write_defaults(self.project_dir / "sdkconfig.defaults")
        idfArgs = [
            "-C",
            self.project_dir,
            "set-target",
            target.name,
        ]
        out = self.invoke_idf_exe(*idfArgs, live=self.print_outputs)
        return out

    def get_idf_cmake_args(self):
        cmake_defs = {"CMAKE_BUILD_TYPE": "Debug" if self.debug else "Release"}
        return [f"-D{key}={value}" for key, value in cmake_defs.items()]

    def compile(self, target, src=None):
        out = ""
        # TODO: build with cmake options
        out += self.prepare(target, src)
        # TODO: support self.num_threads (e.g. patch esp-idf)
        idfArgs = [
            "-C",
            self.project_dir,
            *self.get_idf_cmake_args(),
            "build",
        ]
        out += self.invoke_idf_exe(*idfArgs, live=self.print_outputs)
        return out

    def generate(self, src, target, model=None):
        artifacts = []
        out = self.compile(target, src=src)
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
        metrics = self.get_metrics(elf_file)
        stdout_artifact = Artifact(
            "espidf_out.log", content=out, fmt=ArtifactFormat.TEXT  # TODO: split into one file per command
        )  # TODO: rename to tvmaot_out.log?
        artifacts.append(stdout_artifact)
        return {"default": artifacts}, {"default": metrics}

    def get_idf_serial_args(self, monitor=False):
        args = []
        if self.port:
            args.extend(["-p", self.port])
        if self.baud:
            args.extend(["-B" if monitor else "-b", self.baud])
        return args

    def flash(self, elf, target, timeout=120):
        # Ignore elf, as we use self.project_dir instead
        # TODO: add alternative approach which allows passing elf instead
        if elf is not None:
            logger.debug("Ignoring ELF file for espidf platform")
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

        idfArgs = [
            "-C",
            self.project_dir,
            *self.get_idf_cmake_args(),
            "flash",
            *self.get_idf_serial_args(),
        ]
        self.invoke_idf_exe(*idfArgs, live=self.print_outputs)

    def monitor(self, target, timeout=60):
        if self.flash_only:
            return ""

        if self.use_idf_monitor:

            def _kill_monitor():
                import psutil

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
                env = os.environ.copy()
                env["IDF_PATH"] = str(self.espidf_src_dir)
                env["IDF_TOOLS_PATH"] = str(self.espidf_install_dir)
                cmd = (
                    ". "
                    + str(self.espidf_src_dir / "export.sh")
                    + f"> /dev/null && {self.idf_exe} "
                    + " ".join([str(arg) for arg in args])
                )
                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, executable="/bin/bash", env=env
                )
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
                        exit_code, cmd
                    )
                except KeyboardInterrupt:
                    logger.debug("Interrupted subprocess. Sending SIGINT signal...")
                    _kill_monitor()
                    pid = process.pid
                    os.kill(pid, signal.SIGINT)
                os.system("reset")
                return outStr

            logger.debug("Monitoring target software")
            # TODO: implement timeout
            idfArgs = [
                "-C",
                self.project_dir,
                *self.get_idf_cmake_args(),
                "monitor",
                *self.get_idf_serial_args(monitor=True),
            ]
            return _monitor_helper(
                *idfArgs,
                verbose=self.print_outputs,
                start_match="MLonMCU: START",
                end_match="MLonMCU: STOP",
                timeout=timeout,
            )
        else:
            port = target.port
            if port is None:
                # Falling back to 'espidf.baud'
                assert self.port is not None, f"If using custom serial monitor, please provide '{target.name}.port'"
                port = self.port
            baud = target.baud
            if baud is None:
                # Falling back to 'espidf.baud'
                assert self.baud is not None, f"If using custom serial monitor, please provide '{target.name}.baud'"
                baud = self.baud

            def _monitor_helper2(port, baud, verbose=False, start_match=None, end_match=None, timeout=60):
                # Local import to make this only required for real HW targets
                import serial

                # start_match and end_match are inclusive
                found_start = start_match is None
                outStr = ""
                if timeout:
                    pass  # TODO: implement timeout
                # The following is a custom initialization sequence inspired by
                # (https://github.com/espressif/esp-idf/blob/master/tools/idf_monitor_base/serial_reader.py)
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
            return _monitor_helper2(
                port,
                baud,
                verbose=self.print_outputs,
                start_match="MLonMCU: START",
                end_match="MLonMCU: STOP",
                timeout=timeout,
            )
