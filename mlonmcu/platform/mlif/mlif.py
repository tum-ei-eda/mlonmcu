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
from mlonmcu.timeout import exec_timeout
from mlonmcu.artifact import Artifact, ArtifactFormat
from mlonmcu.logging import get_logger
from mlonmcu.target import get_targets
from mlonmcu.target.target import Target
from mlonmcu.models.utils import get_data_source, fill_data_source_inputs_only

from ..platform import CompilePlatform, TargetPlatform
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
        "fuse_ld": None,
        "strip_strings": False,
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
    }

    REQUIRED = {"mlif.src_dir"}
    OPTIONAL = {"llvm.install_dir"}

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
    def set_inputs(self):
        value = self.config["set_inputs"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def set_inputs_interface(self):
        value = self.config["set_inputs_interface"]
        return value

    @property
    def get_outputs(self):
        value = self.config["get_outputs"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

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
    def template(self):
        return self.config["template"]

    @property
    def template_version(self):
        return self.config["template_version"]

    @property
    def ignore_data(self):
        value = self.config["ignore_data"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def skip_check(self):
        value = self.config["skip_check"]
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
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def debug_symbols(self):
        value = self.config["debug_symbols"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def verbose_makefile(self):
        value = self.config["verbose_makefile"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def lto(self):
        value = self.config["verbose_makefile"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def slim_cpp(self):
        value = self.config["slim_cpp"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def garbage_collect(self):
        value = self.config["garbage_collect"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def fuse_ld(self):
        value = self.config["fuse_ld"]
        return value

    @property
    def strip_strings(self):
        value = self.config["strip_strings"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

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
        if self.toolchain == "llvm" and self.llvm_dir is None:
            raise RuntimeError("Missing config variable: llvm.install_dir")
        else:
            definitions["LLVM_DIR"] = self.llvm_dir
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
        if self.strip_strings is not None:
            definitions["STRIP_STRINGS"] = self.strip_strings

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

    def select_set_inputs_interface(self, target, batch_size):
        in_interface = self.set_inputs_interface
        if in_interface == "auto":
            if target.supports_filesystem:
                in_interface = "filesystem"
            elif target.supports_stdin:
                in_interface = "stdin_raw"
                # TODO: also allow stdin?
            else:  # Fallback
                in_interface = "rom"
        assert in_interface in ["filesystem", "stdin", "stdin_raw", "rom"]
        if batch_size is None:
            if in_interface == "rom":
                batch_size = 1e6  # all inputs are in already compiled into program
            else:
                batch_size = 10
        return in_interface, batch_size

    def select_get_outputs_interface(self, target, batch_size):
        out_interface = self.get_outputs_interface
        if out_interface == "auto":
            if target.supports_filesystem:
                out_interface = "filesystem"
            elif target.supports_stdin:
                out_interface = "stdout_raw"
                # TODO: also allow stdout?
            else:  # Fallback
                out_interface = "ram"
        assert out_interface in ["filesystem", "stdout", "stdout_raw", "ram"]
        if batch_size is None:
            batch_size = 10
        return out_interface, batch_size

    def generate_model_support_code(self, in_interface, out_interface, batch_size):
        code = ""
        code += """
#include "quantize.h"
#include "printing.h"
#include "exit.h"
// #include "ml_interface.h"
#include <cstring>
#include <unistd.h>
#include <stdint.h>
#include <stdbool.h>
#include  <stdio.h>

extern "C" {
int mlif_process_inputs(size_t, bool*);
int mlif_process_outputs(size_t);
void *mlif_input_ptr(int);
void *mlif_output_ptr(int);
int mlif_input_sz(int);
int mlif_output_sz(int);
int mlif_num_inputs();
int mlif_num_outputs();
}
"""
        if in_interface == "rom":
            assert self.inputs_artifact is not None
            import numpy as np
            data = np.load(self.inputs_artifact, allow_pickle=True)
            in_bufs = []
            for i, ins_data in enumerate(data):
                temp = []
                for j, in_data in enumerate(ins_data.values()):
                    byte_data = in_data.tobytes()
                    temp2 = ", ".join(["0x{:02x}".format(x) for x in byte_data] + [""])
                    temp.append(temp2)
                in_bufs.append(temp)

            code += fill_data_source_inputs_only(in_bufs)
            code += """
int mlif_process_inputs(size_t batch_idx, bool *new_)
{
    *new_ = true;
    int num_inputs = mlif_num_inputs();
    for (int i = 0; i < num_inputs; i++)
    {
        int idx = num_inputs * batch_idx + i;
        int size = mlif_input_sz(i);
        char* model_input_ptr = (char*)mlif_input_ptr(i);
        if (idx >= num_data_buffers_in)
        {
            *new_ = false;
            break;
        }
        if (size != data_size_in[idx])
        {
            return EXIT_MLIF_INVALID_SIZE;
        }
        memcpy(model_input_ptr, data_buffers_in[idx], size);
    }
    return 0;
}
"""
        elif in_interface == "stdin_raw":
            code += """
int mlif_process_inputs(size_t batch_idx, bool *new_)
{
    char ch;
    *new_ = true;
    for (int i = 0; i < mlif_num_inputs(); i++)
    {
        int cnt = 0;
        int size = mlif_input_sz(i);
        char* model_input_ptr = (char*)mlif_input_ptr(i);
        while(read(STDIN_FILENO, &ch, 1) > 0) {
            // printf("c=%c / %d\\n", ch, ch);
            model_input_ptr[cnt] = ch;
            cnt++;
            if (cnt == size) {
                break;
            }
        }
        // printf("cnt=%d in_size=%lu\\n", cnt, in_size);
        if (cnt == 0) {
            *new_ = false;
            return 0;
        }
        else if (cnt < size)
        {
            return EXIT_MLIF_INVALID_SIZE;
        }
    }
    return 0;
}
"""
        elif in_interface == "stdin":
            raise NotImplementedError
        elif in_interface == "filesystem":
            raise NotImplementedError
        else:
            assert False
        if out_interface == "ram":
            raise NotImplementedError
        elif out_interface == "stdout_raw":
            # TODO: maybe hardcode num_outputs and size here because we know it
            # and get rid of loop?
            code += """
int mlif_process_outputs(size_t batch_idx)
{
    for (int i = 0; i < mlif_num_outputs(); i++)
    {
        int8_t *model_output_ptr = (int8_t*)mlif_output_ptr(i);
        int size = mlif_output_sz(i);
        // TODO: move markers out of loop
        write(1, "-?-", 3);
        write(1, model_output_ptr, size);
        write(1, "-!-\\n" ,4);
    }
    return 0;
}
"""
        elif out_interface == "stdout":
            raise NotImplementedError
        elif out_interface == "filesystem":
            raise NotImplementedError
        else:
            assert False
        return code

    def generate_model_support(self, target):
        artifacts = []
        in_interface = None
        batch_size = self.batch_size
        if self.set_inputs:
            in_interface, batch_size = self.select_set_inputs_interface(target, batch_size)
        if self.get_outputs:
            out_interface, batch_size = self.select_get_outputs_interface(target, batch_size)
        if in_interface or out_interface:
            code = self.generate_model_support_code(in_interface, out_interface, batch_size)
            code_artifact = Artifact(
                "model_support.cpp", content=code, fmt=ArtifactFormat.TEXT, flags=("model_support"),
            )
            self.definitions["BATCH_SIZE"] = batch_size
            artifacts.append(code_artifact)
        return artifacts

    def configure(self, target, src, _model):
        artifacts = self.generate_model_support(target)
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
            data_artifact = self.gen_data_artifact()
            if data_artifact:
                data_file = self.build_dir / data_artifact.name
                data_artifact.export(data_file)
                cmakeArgs.append("-DDATA_SRC=" + str(data_file))
                artifacts.append(data_artifact)
            else:
                logger.warning("No validation data provided for model.")
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
