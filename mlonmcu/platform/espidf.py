import os
import psutil
import sys
import shutil
import logging
import argparse
import tempfile
import multiprocessing
import subprocess
import distutils.util
from contextlib import closing
from pathlib import Path
from typing import List

from mlonmcu.setup import utils  # TODO: Move one level up?
from mlonmcu.cli.helper.parse import extract_feature_names, extract_config
from mlonmcu.flow import SUPPORTED_BACKENDS, SUPPORTED_FRAMEWORKS
from mlonmcu.target import SUPPORTED_TARGETS
from mlonmcu.config import filter_config
from mlonmcu.feature.features import get_matching_features
from mlonmcu.feature.type import FeatureType
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
        super().__init__("espidf", framework=framework, backend=backend, target=target, features=features, config=config, context=context)
        self.tempdir = None
        self.project_name = "app"
        dir_name = self.name
        # if self.config["project_dir"]:
        if False:
            self.project_dir = Path(self.config["project_dir"])
        else:
            if context:
                assert "temp" in context.environment.paths
                self.project_dir = (
                    context.environment.paths["temp"].path / dir_name
                )  # TODO: Need to lock this for parallel builds
            else:
                logger.info(
                    "Creating temporary directory because no context was available and 'espidf.project_dir' was not supplied"
                )
                self.tempdir = tempfile.TemporaryDirectory()
                self.project_dir = Path(self.tempdir.name) / dir_name
                logger.info("Temporary project directory: %s", self.build_dir)
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
        try:
            subprocess.run([self.idf_exe], shell=True, check=True, stdout=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"It seems like '{self.idf_exe}' is not available. Make sure to setup your environment!") from e


    # def prepare(self, model, ignore_data=False):
    def prepare(self):
        self.check()
        template_dir = self.project_template
        assert template_dir is not None, "No espidf.project_template was provided"  # TODO: fallback to default one?
        template_dir = Path(template_dir)
        assert template_dir.is_dir(), f"Provided project template does not exists: {template_dir}"
        shutil.copytree(template_dir, self.project_dir)
        print("self.project_dir", self.project_dir)
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
        self.prepare()
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
                cmdline =  " ".join(proc.cmdline())
                if "idf_monitor.py" in cmdline: # TODO: do something less "dangerous"?
                    proc.kill()
        def _monitor_helper(*args, verbose=False, start_match=None, end_match=None, timeout=60):
            # start_match and end_match are inclusive
            # TODO: implement timeout
            found_start = (start_match is None)
            logger.debug("- Executing: " + str(args))
            outStr = ""
            process = subprocess.Popen([i for i in args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
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
                            # process.kill()
                            # pid = process.pid
                            # os.kill(pid, signal.SIGINT)
                while exit_code is None:
                    exit_code = process.poll()
                if not verbose and exit_code != 0:
                    logger.error(outStr)
                assert exit_code == 0, "The process returned an non-zero exit code {}! (CMD: `{}`)".format(
                    exit_code, " ".join(list(map(str, args)))
                )
            except KeyboardInterrupt as e:
                logger.debug("Interrupted subprocess. Sending SIGINT signal...")
                _kill_monitor()
                pid = process.pid
                os.kill(pid, signal.SIGINT)
            print("outStr", outStr)
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
        return _monitor_helper(*idfArgs, verbose=self.print_output, start_match="MLonMCU: START", end_match="MLonMCU: STOP", timeout=timeout)
