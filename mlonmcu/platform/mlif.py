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
"""MLIF Platform"""
import tempfile

from pathlib import Path

from mlonmcu.setup import utils  # TODO: Move one level up?
from mlonmcu.artifact import Artifact, ArtifactFormat
from mlonmcu.logging import get_logger

from .platform import CompilePlatform

logger = get_logger()


class MlifPlatform(CompilePlatform):
    """Model Library Interface Platform class."""

    FEATURES = CompilePlatform.FEATURES + ["validate"]

    DEFAULTS = {
        **CompilePlatform.DEFAULTS,
        "ignore_data": True,
        "model_support_dir": None,
        "toolchain": "gcc",
        "prebuild_lib_path": None,
    }

    REQUIRED = []

    def __init__(self, framework, backend, target, features=None, config=None, context=None):
        super().__init__(
            "mlif",
            framework=framework,
            backend=backend,
            target=target,
            features=features,
            config=config,
            context=context,
        )
        self.goal = "generic_mlif"
        flags = [self.framework.name, self.backend.name, self.target.name] + [feature.name for feature in self.features]
        dir_name = utils.makeDirName("mlif", flags=flags)
        self.tempdir = None
        if self.config["build_dir"]:
            self.build_dir = Path(self.config["build_dir"])
        else:
            if context:
                assert "temp" in context.environment.paths
                self.build_dir = (
                    context.environment.paths["temp"].path / dir_name
                )  # TODO: Need to lock this for parallel builds
            else:
                logger.info(
                    "Creating temporary directory because no context was available"
                    "and 'mlif.build_dir' was not supplied"
                )
                self.tempdir = tempfile.TemporaryDirectory()
                self.build_dir = Path(self.tempdir.name) / dir_name
                logger.info("Temporary build directory: %s", self.build_dir)
        if "mlif.src_dir" in self.config:
            if self.context:
                logger.warning("User has overwritten the value of 'mlif.src_dir'")
            self.mlif_dir = Path(self.config["mlif.src_dir"])
        else:
            if context:
                self.mlif_dir = Path(context.environment.home) / "sw"  # TODO: Define in env paths
            else:
                raise RuntimeError("Please define the value of 'mlif.src_dir' or pass a context")

    def set_directory(self, directory):
        self.build_dir = directory

    @property
    def ignore_data(self):
        return bool(self.config["ignore_data"])

    @property
    def toolchain(self):
        return str(self.config["toolchain"])

    @property
    def model_support_dir(self):
        return self.config["model_support_dir"]

    @property
    def prebuild_lib_dir(self):
        return self.config["prebuild_lib_dir"]

    def close(self):
        if self.tempdir:
            self.tempdir.cleanup()

    def get_common_cmake_args(self, num=1):
        args = []
        args.append(f"-DNUM_RUNS={num}")
        args.append(f"-DTOOLCHAIN={self.toolchain}")
        return args

    # def prepare(self, model, ignore_data=False):
    def prepare(self):
        # utils.mkdirs(self.data_dir)
        # data_file = self.data_dir / "data.c"
        # write_inout_data(model, data_file, skip=ignore_data)
        # return data_file
        pass

    def configure(self, src, _model, num=1, data_file=None):
        if not isinstance(src, Path):
            src = Path(src)
        cmakeArgs = []
        cmakeArgs.extend(self.framework.get_cmake_args())
        cmakeArgs.extend(self.backend.get_cmake_args())
        cmakeArgs.extend(self.target.get_cmake_args())
        cmakeArgs.extend(self.get_common_cmake_args(num=num))
        if self.model_support_dir:
            cmakeArgs.append(f"-DMODEL_SUPPORT_DIR={self.model_support_dir}")
        else:
            pass
            # args.append(f"-DMODEL_DIR={?}")
        if src.is_file():
            src = src.parent  # TODO deal with directories or files?
        if src.is_dir():
            cmakeArgs.append("-DSRC_DIR=" + str(src))
        else:
            raise RuntimeError("Unable to find sources!")
        # data_file = self.prepare(model, ignore_data=(not debug or self.ignore_data))
        if self.ignore_data:
            cmakeArgs.append("-DDATA_SRC=")
        else:
            assert data_file is not None, "No data.c file was supplied"
            cmakeArgs.append("-DDATA_SRC=" + str(data_file))
        utils.mkdirs(self.build_dir)
        utils.cmake(
            self.mlif_dir,
            *cmakeArgs,
            cwd=self.build_dir,
            debug=self.debug,
            live=self.print_output,
        )

    def compile(self, src=None, model=None, num=1, data_file=None):
        if src:
            self.configure(src, model, num=num, data_file=data_file)
        utils.make(
            self.goal,
            cwd=self.build_dir,
            threads=self.num_threads,
            live=self.print_output,
        )

    def generate_elf(self, src=None, model=None, num=1, data_file=None):
        artifacts = []
        self.compile(src=src, model=model, num=num, data_file=data_file)
        elf_file = self.build_dir / "bin" / "generic_mlif"
        # TODO: just use path instead of raw data?
        with open(elf_file, "rb") as handle:
            data = handle.read()
            artifact = Artifact("generic_mlif", raw=data, fmt=ArtifactFormat.RAW)
            artifacts.append(artifact)
        self.artifacts = artifacts
