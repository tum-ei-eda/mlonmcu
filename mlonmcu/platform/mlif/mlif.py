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
from typing import Tuple

from pathlib import Path

from mlonmcu.config import str2bool
from mlonmcu.setup import utils  # TODO: Move one level up?
from mlonmcu.artifact import Artifact, ArtifactFormat
from mlonmcu.logging import get_logger
from mlonmcu.target import get_targets
from mlonmcu.target.target import Target
from mlonmcu.models.utils import get_data_source

from ..platform import CompilePlatform, TargetPlatform
from .mlif_target import get_mlif_platform_targets, create_mlif_platform_target

logger = get_logger()


class MlifPlatform(CompilePlatform, TargetPlatform):
    """Model Library Interface Platform class."""

    FEATURES = (
        CompilePlatform.FEATURES
        + TargetPlatform.FEATURES
        + [
            "validate",
            "muriscvnn",
            "cmsisnn",
            "muriscvnnbyoc",
            "cmsisnnbyoc",
            "vext",
            "pext",
            "arm_mvei",
            "arm_dsp",
            "auto_vectorize",
            "benchmark",
            "xpulp",
        ]  # TODO: allow Feature-Features with automatic resolution of initialization order
    )

    DEFAULTS = {
        **CompilePlatform.DEFAULTS,
        **TargetPlatform.DEFAULTS,
        "ignore_data": True,
        "fail_on_error": False,  # Prefer to add acolum with validation results instead of raising a RuntimeError
        "model_support_dir": None,
        "toolchain": "gcc",
        "prebuild_lib_path": None,
        "optimize": None,  # values: 0,1,2,3,s
        "input_data_path": None,
        "output_data_path": None,
        "mem_only": False,
        "debug_symbols": False,
        "verbose_makefile": False,
    }

    REQUIRED = ["mlif.src_dir"]
    OPTIONAL = ["llvm.install_dir"]

    def __init__(self, features=None, config=None):
        super().__init__(
            "mlif",
            features=features,
            config=config,
        )
        self.tempdir = None
        self.build_dir = None
        self.goal = "generic_mlif"

    def gen_data_artifact(self):
        in_paths = self.input_data_path
        if not isinstance(in_paths, list):
            in_paths = [in_paths]
        in_paths_new = []
        for in_path in in_paths:
            if in_path.is_file():
                raise NotImplementedError
            elif in_path.is_dir():
                in_paths_new.extend([f for f in Path(in_path).iterdir() if f.is_file()])
            else:
                logger.warning("TODO")
                return None
        in_paths = in_paths_new
        out_paths = self.output_data_path
        if not isinstance(out_paths, list):
            out_paths = [out_paths]
        out_paths_new = []
        for out_path in out_paths:
            if out_path.is_file():
                raise NotImplementedError
            elif out_path.is_dir():
                out_paths_new.extend([f for f in Path(out_path).iterdir() if f.is_file()])
            else:
                logger.warning("TODO")
                return None
        out_paths = out_paths_new
        if len(in_paths) == 0 or len(out_paths) == 0:
            logger.warning("TODO")
            return None
        data_src = get_data_source(in_paths, out_paths)
        return Artifact("data.c", content=data_src, fmt=ArtifactFormat.SOURCE)

    def init_directory(self, path=None, context=None):
        if self.build_dir is not None:
            self.build_dir.mkdir(exist_ok=True)
            logger.debug("Build directory already initialized")
            return
        dir_name = self.name
        if path is not None:
            self.build_dir = Path(path)
        elif self.config["build_dir"]:
            self.build_dir = Path(self.config["build_dir"])
        else:
            if context:
                assert "temp" in context.environment.paths
                self.build_dir = (
                    context.environment.paths["temp"].path / dir_name
                )  # TODO: Need to lock this for parallel builds
            else:
                logger.debug(
                    "Creating temporary directory because no context was available "
                    "and 'mlif.build_dir' was not supplied"
                )
                self.tempdir = tempfile.TemporaryDirectory()
                self.build_dir = Path(self.tempdir.name) / dir_name
                logger.info("Temporary build directory: %s", self.build_dir)
        self.build_dir.mkdir(exist_ok=True)

    def create_target(self, name):
        assert name in self.get_supported_targets(), f"{name} is not a valid MLIF target"
        targets = get_targets()
        if name in targets:
            base = targets[name]
        else:
            base = Target
        return create_mlif_platform_target(name, self, base=base)

    @property
    def mlif_dir(self):
        return Path(self.config["mlif.src_dir"])

    @property
    def llvm_dir(self):
        return self.config["llvm.install_dir"]

    @property
    def ignore_data(self):
        value = self.config["ignore_data"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def fail_on_error(self):
        value = self.config["fail_on_error"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def validate_outputs(self):
        return not self.ignore_data

    @property
    def toolchain(self):
        return str(self.config["toolchain"])

    @property
    def model_support_dir(self):
        return self.config["model_support_dir"]

    @property
    def prebuild_lib_dir(self):
        return self.config["prebuild_lib_dir"]

    @property
    def optimize(self):
        return self.config["optimize"]

    @property
    def input_data_path(self):
        return self.config["input_data_path"]

    @property
    def output_data_path(self):
        return self.config["output_data_path"]

    @property
    def mem_only(self):
        value = self.config["mem_only"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def debug_symbols(self):
        value = self.config["debug_symbols"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def verbose_makefile(self):
        value = self.config["verbose_makefile"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    def get_supported_targets(self):
        target_names = get_mlif_platform_targets()
        return target_names

    def close(self):
        if self.tempdir:
            self.tempdir.cleanup()

    def get_common_cmake_args(self):
        args = []
        args.append(f"-DTOOLCHAIN={self.toolchain}")
        if self.toolchain == "llvm" and self.llvm_dir is None:
            raise RuntimeError("Missing config variable: llvm.install_dir")
        else:
            args.append(f"-DLLVM_DIR={self.llvm_dir}")
        if self.optimize:
            args.append(f"-DOPTIMIZE={self.optimize}")
        if self.debug_symbols:
            args.append("-DDEBUG_SYMBOLS=ON")
        if self.verbose_makefile:
            args.append("-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON")
        if self.model_support_dir:
            args.append(f"-DMODEL_SUPPORT_DIR={self.model_support_dir}")
        else:
            pass
        return args

    def prepare(self):
        self.init_directory()

    def configure(self, target, src, _model):
        del target
        if not isinstance(src, Path):
            src = Path(src)
        cmakeArgs = []
        definitions = self.definitions
        if self.mem_only:
            definitions["QUIET"] = True
        for key, value in definitions.items():
            if isinstance(value, bool):
                value = "ON" if value else "OFF"
            cmakeArgs.append(f"-D{key}={value}")
        cmakeArgs.extend(self.get_common_cmake_args())
        if src.is_file():
            src = src.parent  # TODO deal with directories or files?
        if src.is_dir():
            cmakeArgs.append("-DSRC_DIR=" + str(src))
        else:
            raise RuntimeError("Unable to find sources!")
        if self.ignore_data:
            cmakeArgs.append("-DDATA_SRC=")
            artifacts = []
        else:
            data_artifact = self.gen_data_artifact()
            data_file = self.build_dir / data_artifact.name
            data_artifact.export(data_file)
            cmakeArgs.append("-DDATA_SRC=" + str(data_file))
            artifacts = [data_artifact]
        utils.mkdirs(self.build_dir)
        out = utils.cmake(
            self.mlif_dir,
            *cmakeArgs,
            cwd=self.build_dir,
            debug=self.debug,
            live=self.print_outputs,
        )
        return out, artifacts

    def compile(self, target, src=None, model=None, data_file=None):
        out = ""
        if src:
            configure_out, artifacts = self.configure(target, src, model)
            out += configure_out
        out += utils.make(
            self.goal,
            cwd=self.build_dir,
            threads=self.num_threads,
            live=self.print_outputs,
        )
        return out, artifacts

    def generate(self, src, target, model=None) -> Tuple[dict, dict]:
        out, artifacts = self.compile(target, src=src, model=model)
        elf_file = self.build_dir / "bin" / "generic_mlif"
        # TODO: just use path instead of raw data?
        with open(elf_file, "rb") as handle:
            data = handle.read()
            artifact = Artifact("generic_mlif", raw=data, fmt=ArtifactFormat.RAW)
            artifacts.insert(0, artifact)  # First artifact should be the ELF
        metrics = self.get_metrics(elf_file)
        stdout_artifact = Artifact(
            "mlif_out.log", content=out, fmt=ArtifactFormat.TEXT
        )  # TODO: rename to tvmaot_out.log?
        artifacts.append(stdout_artifact)
        return {"default": artifacts}, {"default": metrics}
