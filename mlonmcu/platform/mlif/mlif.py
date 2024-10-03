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
import os
import tempfile
from typing import Tuple
from pathlib import Path

import yaml
import numpy as np

from mlonmcu.config import str2bool
from mlonmcu.setup import utils  # TODO: Move one level up?
from mlonmcu.timeout import exec_timeout
from mlonmcu.artifact import Artifact, ArtifactFormat
from mlonmcu.logging import get_logger
from mlonmcu.target import get_targets
from mlonmcu.target.target import Target
from mlonmcu.models.utils import get_data_source

from ..platform import CompilePlatform, TargetPlatform
from .interfaces import ModelSupport
from .mlif_target import get_mlif_platform_targets, create_mlif_platform_target

logger = get_logger()


class MlifPlatform(CompilePlatform, TargetPlatform):
    """Model Library Interface Platform class."""

    FEATURES = (
        CompilePlatform.FEATURES
        | TargetPlatform.FEATURES
        | {
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
            "set_inputs",
            "get_outputs",
        }  # TODO: allow Feature-Features with automatic resolution of initialization order
    )

    DEFAULTS = {
        **CompilePlatform.DEFAULTS,
        **TargetPlatform.DEFAULTS,
        "template": "ml_interface",
        "template_version": None,
        "ignore_data": True,
        "skip_check": False,
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
        "lto": False,
        "slim_cpp": True,
        "garbage_collect": True,
        "strip_strings": False,
        "unroll_loops": None,
        "goal": "generic_mlonmcu",  # Use 'generic_mlif' for older version of MLIF
        "set_inputs": False,
        "set_inputs_interface": None,
        "get_outputs": False,
        "get_outputs_interface": None,
        "get_outputs_fmt": None,
        "batch_size": None,
        "model_support_file": None,
        "model_support_dir": None,
        "model_support_lib": None,
        # llvm specific (TODO: move to toolchain components)
        "fuse_ld": None,
        "global_isel": False,
        "extend_attrs": False,
        "ccache": False,
    }

    REQUIRED = {"mlif.src_dir"}
    OPTIONAL = {"llvm.install_dir", "srecord.install_dir"}

    def __init__(self, features=None, config=None):
        super().__init__(
            "mlif",
            features=features,
            config=config,
        )
        self.tempdir = None
        self.build_dir = None

    @property
    def goal(self):
        return self.config["goal"]

    @property
    def ccache(self):
        value = self.config["ccache"]
        return str2bool(value)

    @property
    def set_inputs(self):
        value = self.config["set_inputs"]
        return str2bool(value)

    @property
    def set_inputs_interface(self):
        value = self.config["set_inputs_interface"]
        return value

    @property
    def get_outputs(self):
        value = self.config["get_outputs"]
        return str2bool(value)

    @property
    def get_outputs_interface(self):
        value = self.config["get_outputs_interface"]
        return value

    @property
    def get_outputs_fmt(self):
        value = self.config["get_outputs_fmt"]  # TODO: use
        return value

    @property
    def batch_size(self):
        value = self.config["batch_size"]  # TODO: use
        if isinstance(value, str):
            value = int(value)
        return value

    @property
    def inputs_artifact(self):
        # THIS IS A HACK (get inputs fom artifacts!)
        lookup_path = self.build_dir.parent / "inputs.npy"
        if lookup_path.is_file():
            return lookup_path
        else:
            logger.warning("Artifact 'inputs.npz' not found!")
            return None

    @property
    def model_info_file(self):
        # THIS IS A HACK (get inputs fom artifacts!)
        lookup_path = self.build_dir.parent / "model_info.yml"
        if lookup_path.is_file():
            return lookup_path
        else:
            logger.warning("Artifact 'model_info.yml' not found!")
            return None

    @property
    def needs_model_support(self):
        return self.set_inputs or self.get_outputs

    def gen_data_artifact(self):
        in_paths = self.input_data_path
        if not isinstance(in_paths, list):
            in_paths = [in_paths]
        in_paths_new = []
        for in_path in in_paths:
            if in_path is None:
                continue
            if not isinstance(in_path, Path):
                in_path = Path(in_path)
            if in_path.is_file():
                raise NotImplementedError
            elif in_path.is_dir():
                in_paths_new.extend([f for f in in_path.iterdir() if f.is_file()])
            else:
                return None
        in_paths = in_paths_new
        out_paths = self.output_data_path
        if not isinstance(out_paths, list):
            out_paths = [out_paths]
        out_paths_new = []
        for out_path in out_paths:
            if out_path is None:
                continue
            if not isinstance(out_path, Path):
                out_path = Path(out_path)
            if out_path.is_file():
                raise NotImplementedError
            elif out_path.is_dir():
                out_paths_new.extend([f for f in out_path.iterdir() if f.is_file()])
            else:
                return None
        out_paths = out_paths_new
        if len(in_paths) == 0 or len(out_paths) == 0:
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
    def srecord_dir(self):
        return self.config["srecord.install_dir"]

    @property
    def template(self):
        return self.config["template"]

    @property
    def template_version(self):
        return self.config["template_version"]

    @property
    def ignore_data(self):
        value = self.config["ignore_data"]
        return str2bool(value)

    @property
    def skip_check(self):
        value = self.config["skip_check"]
        return str2bool(value)

    @property
    def fail_on_error(self):
        value = self.config["fail_on_error"]
        return str2bool(value)

    @property
    def validate_outputs(self):
        return not self.ignore_data

    @property
    def toolchain(self):
        return str(self.config["toolchain"])

    @property
    def model_support_file(self):
        value = self.config["model_support_file"]  # TODO: use
        return value

    @property
    def model_support_dir(self):
        value = self.config["model_support_dir"]  # TODO: use
        return value

    @property
    def model_support_lib(self):
        value = self.config["model_support_lib"]  # TODO: use
        return value

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
        return str2bool(value)

    @property
    def debug_symbols(self):
        value = self.config["debug_symbols"]
        return str2bool(value)

    @property
    def verbose_makefile(self):
        value = self.config["verbose_makefile"]
        return str2bool(value)

    @property
    def lto(self):
        value = self.config["lto"]
        return str2bool(value)

    @property
    def slim_cpp(self):
        value = self.config["slim_cpp"]
        return str2bool(value)

    @property
    def garbage_collect(self):
        value = self.config["garbage_collect"]
        return str2bool(value)

    @property
    def fuse_ld(self):
        value = self.config["fuse_ld"]
        return value

    @property
    def global_isel(self):
        value = self.config["global_isel"]
        return str2bool(value)

    @property
    def extend_attrs(self):
        value = self.config["extend_attrs"]
        return str2bool(value)

    @property
    def strip_strings(self):
        value = self.config["strip_strings"]
        return str2bool(value)

    @property
    def unroll_loops(self):
        value = self.config["unroll_loops"]
        return str2bool(value, allow_none=True)

    def get_supported_targets(self):
        target_names = get_mlif_platform_targets()
        return target_names

    def close(self):
        if self.tempdir:
            self.tempdir.cleanup()

    def get_definitions(self):
        definitions = self.definitions
        definitions["TEMPLATE"] = self.template
        if self.template_version:
            definitions["TEMPLATE_VERSION"] = self.template_version
        definitions["TOOLCHAIN"] = self.toolchain
        definitions["QUIET"] = self.mem_only
        definitions["SKIP_CHECK"] = self.skip_check
        if self.batch_size is not None:
            definitions["BATCH_SIZE"] = self.batch_size
        if self.num_threads is not None:
            definitions["SUBPROJECT_THREADS"] = self.num_threads
        if self.toolchain == "llvm":
            if self.llvm_dir is None:
                raise RuntimeError("Missing config variable: llvm.install_dir")
            llvm_dir = Path(self.llvm_dir).resolve()
            assert llvm_dir.is_dir(), f"llvm.install_dir does not exist: {llvm_dir}"
            definitions["LLVM_DIR"] = llvm_dir
        if self.optimize is not None:
            definitions["OPTIMIZE"] = self.optimize
        if self.debug_symbols is not None:
            definitions["DEBUG_SYMBOLS"] = self.debug_symbols
        if self.verbose_makefile is not None:
            definitions["CMAKE_VERBOSE_MAKEFILE"] = self.verbose_makefile
        if self.lto is not None:
            definitions["ENABLE_LTO"] = self.lto
        if self.garbage_collect is not None:
            definitions["ENABLE_GC"] = self.garbage_collect
        if self.slim_cpp is not None:
            definitions["SLIM_CPP"] = self.slim_cpp
        if self.model_support_file is not None:
            definitions["MODEL_SUPPORT_FILE"] = self.model_support_file
        if self.model_support_dir is not None:
            definitions["MODEL_SUPPORT_DIR"] = self.model_support_dir
        if self.model_support_lib is not None:
            definitions["MODEL_SUPPORT_LIB"] = self.model_support_lib
        if self.fuse_ld is not None:
            definitions["FUSE_LD"] = self.fuse_ld
        if self.global_isel is not None:
            definitions["GLOBAL_ISEL"] = self.global_isel
        if self.extend_attrs is not None:
            definitions["EXTEND_ATTRS"] = self.extend_attrs
        if self.strip_strings is not None:
            definitions["STRIP_STRINGS"] = self.strip_strings
        if self.unroll_loops is not None:
            definitions["UNROLL_LOOPS"] = self.unroll_loops
        if self.ccache:
            definitions["CMAKE_C_COMPILER_LAUNCHER"] = "ccache"  # TODO: choose between ccache/sccache
            definitions["CMAKE_CXX_COMPILER_LAUNCHER"] = "ccache"  # TODO: choose between ccache/sccache

        return definitions

    def get_cmake_args(self):
        cmakeArgs = []
        definitions = self.get_definitions()
        for key, value in definitions.items():
            if isinstance(value, bool):
                value = "ON" if value else "OFF"
            cmakeArgs.append(f"-D{key}={value}")
        return cmakeArgs

    def prepare(self):
        self.init_directory()

    def prepare_environment(self):
        env = os.environ.copy()
        if self.srecord_dir:
            path_old = env["PATH"]
            path_new = f"{self.srecord_dir}:{path_old}"
            env["PATH"] = path_new
        return env

    def generate_model_support(self, target):
        artifacts = []
        batch_size = self.batch_size
        inputs_data = None
        if self.inputs_artifact is not None:
            inputs_data = np.load(self.inputs_artifact, allow_pickle=True)
        if self.model_info_file is not None:
            with open(self.model_info_file, "r") as f:
                model_info = yaml.safe_load(f)
        if self.set_inputs or self.get_outputs:
            model_support = ModelSupport(
                in_interface=self.set_inputs_interface,
                out_interface=self.get_outputs_interface,
                model_info=model_info,
                target=target,
                batch_size=batch_size,
                inputs_data=inputs_data,
            )
            code = model_support.generate()
            code_artifact = Artifact(
                "model_support.cpp",
                content=code,
                fmt=ArtifactFormat.TEXT,
                flags=("model_support"),
            )
            self.definitions["BATCH_SIZE"] = model_support.batch_size
            artifacts.append(code_artifact)
        return artifacts

    def configure(self, target, src, _model):
        artifacts = []
        if self.needs_model_support:
            artifacts.extend(self.generate_model_support(target))
            if len(artifacts) > 0:
                assert len(artifacts) == 1
                model_support_artifact = artifacts[0]
                model_support_file = self.build_dir / model_support_artifact.name
                model_support_artifact.export(model_support_file)
                self.definitions["MODEL_SUPPORT_FILE"] = model_support_file
            del target
        if not isinstance(src, Path):
            src = Path(src)
        cmakeArgs = self.get_cmake_args()
        if src.is_file():
            src = src.parent  # TODO deal with directories or files?
        if src.is_dir():
            cmakeArgs.append("-DSRC_DIR=" + str(src))
        else:
            raise RuntimeError("Unable to find sources!")
        if self.ignore_data:
            cmakeArgs.append("-DDATA_SRC=")
        else:
            # data_artifact = self.gen_data_artifact()
            data_artifact = None
            if data_artifact:
                data_file = self.build_dir / data_artifact.name
                data_artifact.export(data_file)
                cmakeArgs.append("-DDATA_SRC=" + str(data_file))
                artifacts.append(data_artifact)
            else:
                logger.warning("No validation data provided for model.")
        utils.mkdirs(self.build_dir)
        env = self.prepare_environment()
        out = utils.cmake(
            self.mlif_dir,
            *cmakeArgs,
            cwd=self.build_dir,
            debug=self.debug,
            live=self.print_outputs,
            env=env,
        )
        return out, artifacts

    def compile(self, target, src=None, model=None, data_file=None):
        out = ""
        if src:
            configure_out, artifacts = self.configure(target, src, model)
            out += configure_out
        env = self.prepare_environment()
        out += utils.make(
            self.goal,
            cwd=self.build_dir,
            threads=self.num_threads,
            live=self.print_outputs,
            env=env,
        )
        return out, artifacts

    def generate(self, src, target, model=None) -> Tuple[dict, dict]:
        # TODO: fix timeouts
        if self.validate_outputs:
            # some strange bug?
            self.timeout_sec = 0
        else:
            self.timeout_sec = 90
        self.timeout_sec = 0
        if self.timeout_sec > 0:
            out, artifacts = exec_timeout(
                self.timeout_sec,
                self.compile,
                target,
                src=src,
                model=model,
            )
        else:
            out, artifacts = self.compile(target, src=src, model=model)
        elf_file = self.build_dir / "bin" / self.goal
        map_file = self.build_dir / "linker.map"  # TODO: optional
        hex_file = self.build_dir / "bin" / "generic_mlonmcu.hex"
        path_file = self.build_dir / "bin" / "generic_mlonmcu.path"  # TODO: move to dumps
        asmdump_file = self.build_dir / "dumps" / "generic_mlonmcu.dump"  # TODO: optional
        srcdump_file = self.build_dir / "dumps" / "generic_mlonmcu.srcdump"  # TODO: optional

        # TODO: just use path instead of raw data?
        with open(elf_file, "rb") as handle:
            data = handle.read()
            artifact = Artifact("generic_mlonmcu", raw=data, fmt=ArtifactFormat.RAW)
            artifacts.insert(0, artifact)  # First artifact should be the ELF
        # for cv32e40p
        if hex_file.is_file():
            with open(hex_file, "rb") as handle:
                data = handle.read()
                artifact = Artifact("generic_mlonmcu.hex", raw=data, fmt=ArtifactFormat.RAW)
                artifacts.insert(1, artifact)
        # only for vicuna
        if path_file.is_file():
            with open(path_file, "r") as handle:
                data = handle.read()
                artifact = Artifact("generic_mlonmcu.path", content=data, fmt=ArtifactFormat.TEXT)
                artifacts.insert(1, artifact)
        if map_file.is_file():
            with open(map_file, "r") as handle:
                data = handle.read()
                artifact = Artifact("generic_mlonmcu.map", content=data, fmt=ArtifactFormat.TEXT)
                artifacts.append(artifact)
        if asmdump_file.is_file():
            with open(asmdump_file, "r") as handle:
                data = handle.read()
                artifact = Artifact(
                    "generic_mlonmcu.dump", content=data, fmt=ArtifactFormat.TEXT, flags=(self.toolchain,)
                )
                artifacts.append(artifact)
        if srcdump_file.is_file():
            with open(srcdump_file, "r") as handle:
                data = handle.read()
                artifact = Artifact(
                    "generic_mlonmcu.srcdump", content=data, fmt=ArtifactFormat.TEXT, flags=(self.toolchain,)
                )
                artifacts.append(artifact)
        metrics = self.get_metrics(elf_file)
        stdout_artifact = Artifact(
            "mlif_out.log", content=out, fmt=ArtifactFormat.TEXT
        )  # TODO: rename to tvmaot_out.log?
        artifacts.append(stdout_artifact)
        return {"default": artifacts}, {"default": metrics}
