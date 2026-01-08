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
"""CFU Playground Platform"""

import os
import shutil
import tempfile
from pathlib import Path
import pkg_resources


from mlonmcu.setup import utils
from mlonmcu.artifact import Artifact, ArtifactFormat
from mlonmcu.logging import get_logger
from mlonmcu.flow.tvm.framework import get_crt_config_dir

from ..platform import CompilePlatform, TargetPlatform
from .cfu_playground_target import create_cfu_playground_platform_target, get_cfu_playground_platform_targets

logger = get_logger()


def get_project_template(name="proj_template_no_tflm"):
    cfu_templates = pkg_resources.resource_listdir("mlonmcu", os.path.join("resources", "platforms", "cfu_playground"))
    if name not in cfu_templates:
        return None
    fname = pkg_resources.resource_filename("mlonmcu", os.path.join("resources", "platforms", "cfu_playground", name))
    return fname


class CFUPlaygroundPlatform(CompilePlatform, TargetPlatform):
    """CFU Playground Platform class."""

    FEATURES = CompilePlatform.FEATURES | TargetPlatform.FEATURES | {"benchmark"}

    DEFAULTS = {
        **CompilePlatform.DEFAULTS,
        **TargetPlatform.DEFAULTS,
        "project_template": None,
        "project_dir": None,
        "optimize": None,  # values: 0,1,2,3,s
        "mlif_template": None,
        # "device": "digilent_arty",
        # "use_renode": True,
        # "use_verilator": True,
    }

    REQUIRED = {
        "cfu_playground.src_dir",
        "yosys.install_dir",
        "mlif.src_dir",
    }  # TODO: yosys, riscv tc?
    OPTIONAL = {"tvm.src_dir", "mlif.template"}

    def __init__(self, features=None, config=None):
        super().__init__(
            "cfu_playground",
            features=features,
            config=config,
        )
        self.tempdir = None
        self.project_name = "app"
        self.project_dir = None

    @property
    def cfu_playground_src_dir(self):
        return Path(self.config["cfu_playground.src_dir"])

    @property
    def mlif_src_dir(self):
        return Path(self.config["mlif.src_dir"])

    @property
    def tvm_src_dir(self):
        return Path(self.config["tvm.src_dir"])

    @property
    def yosys_install_dir(self):
        return Path(self.config["yosys.install_dir"])

    @property
    def mlif_template(self):
        value = self.config["mlif_template"]
        value2 = self.config["mlif.template"]
        if value is None:
            return None
        return Path(value)

    @property
    def use_renode(self):
        return True

    def init_directory(self, path=None, context=None):
        if self.project_dir is not None:
            self.project_dir.mkdir(exist_ok=True)
            logger.debug("Project directory already initialized")
            return self.project_dir
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
                    "and 'cfu_playground.project_dir' was not supplied"
                )
                self.tempdir = tempfile.TemporaryDirectory()
                self.project_dir = Path(self.tempdir.name) / dir_name
                logger.debug("Temporary project directory: %s", self.project_dir)
        self.project_dir.mkdir(exist_ok=True)
        return self.project_dir

    def _get_supported_targets(self):
        print("_get_supported_targets")
        ret = get_cfu_playground_platform_targets()
        print("ret", ret)
        return ret

    def create_target(self, name):
        supported = self.get_supported_targets()
        assert name in supported, f"{name} is not a valid CFU Playground device"
        base = supported[name]
        return create_cfu_playground_platform_target(name, self, base=base)

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
    def optimize(self):
        val = self.config["optimize"]
        if val is None:
            val = "s"
        else:
            val = str(val)
        assert val in ["0", "g", "2", "s", "z"], f"Unsupported: {val}"
        return val

    def close(self):
        if self.tempdir:
            self.tempdir.cleanup()

    def check(self):
        pass

    def get_backend(self):
        return self.definitions.get("MLONMCU_BACKEND")

    def get_framework(self):
        return self.definitions.get("MLONMCU_FRAMEWORK")

    # TODO: get TVM_CRT_CONFIG_DIR, TVM_DIR,... from definitions?

    def get_makefile_defines(self):
        # definitions = self.definitions
        defines = {}
        if not self.debug:
            defines["NDEBUG"] = None

        print("defines", defines)
        return defines

    def get_makefile_exports(self):
        exports = self.definitions
        exports["TEMPLATE"] = self.mlif_template
        # if self.template_version:
        #     definitions["TEMPLATE_VERSION"] = self.mlif_template_version
        # definitions["TOOLCHAIN"] = self.toolchain
        if self.optimize is not None:
            exports["OPTIMIZE"] = self.optimize

        print("exports", exports)
        return exports

    def get_makefile_includes(self):
        includes = []

        print("includes", includes)
        return includes

    def prepare_environment(self, target=None):
        env = os.environ.copy()
        # TODO: riscv tc from target
        # TODO: yosys
        # TODO: ...
        env["CFU_ROOT"] = self.cfu_playground_src_dir
        env["PROJ"] = "cfu_playground"
        env["PROJ_DIR"] = self.project_dir
        new_path = env.get("PATH", "")
        # new_path = f"{self.yosys_install_dir}/bin:{old_path}"
        new_path = f"{self.yosys_install_dir}:{new_path}"
        if target:
            new_path = f"{target.riscv_gcc_prefix}/bin:{new_path}"
        print("new_path", new_path)
        env["PATH"] = new_path
        return env

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
        # print("cp", template_dir, self.project_dir)
        # TODO: pass backend to platform?
        backend = self.get_backend()
        framework = self.get_framework()
        print("backend", backend)
        print("framework", framework)
        if (src / "dummy_model").is_file():
            assert self.mlif_template is not None, "Undefined cfu_playground.mlif_template"
            backend = "none"
        print("backend2", backend)
        # elif (src / "aot_wrapper.c").is_file():
        #     backend = "tvmaot"
        # elif (src / "rt_wrapper.c").is_file():
        #     backend = "tvmrt"
        # elif (src / "model.cc.h").is_file():
        #     backend = "tflmi"
        assert backend is not None, "Could not infer used backend"
        shutil.copytree(template_dir, self.project_dir, dirs_exist_ok=True)
        print("src", src)
        dest_base = self.project_dir / "src"
        makefile_exports = self.get_makefile_exports()
        makefile_defines = self.get_makefile_defines()
        makefile_includes = self.get_makefile_includes()
        makefile_includes += ["test_inc_dir"]
        if backend == "none":
            to_copy = []
            mlif_template_dir = self.mlif_template
            if not self.mlif_template.is_dir():
                mlif_template_dir = self.mlif_src_dir / "lib" / self.mlif_template
            print("mlif_template_dir", mlif_template_dir)
            assert mlif_template_dir.is_dir()
            bench_name = "hello_world"  # TODO: expose
            to_copy += [(mlif_template_dir / f"{bench_name}.c", dest_base)]

        elif backend in ["tvmaot", "tvmaotplus", "tvmrt", "tvmllvm"]:
            crt_config_dir = Path(get_crt_config_dir())
            assert crt_config_dir.is_dir()
            to_copy = [
                (crt_config_dir, dest_base),
                (src / "tvm_wrapper.h", dest_base),
                (src / "codegen" / "host" / "src", dest_base),
                (self.mlif_src_dir / "lib/ml_interface/v1/default_data.c", dest_base),
                (self.mlif_src_dir / "lib/ml_interface/v1/ml_interface.h", dest_base),
                (self.mlif_src_dir / "lib/ml_interface/v1/ml_interface.c", dest_base),
                (self.mlif_src_dir / "lib/ml_interface/v1/ml_interface_tvm.c", dest_base),
                (self.mlif_src_dir / "lib/ml_interface/v1/default_model_support/process_input.c", dest_base),
                (self.mlif_src_dir / "lib/ml_interface/v1/default_model_support/process_output.c", dest_base),
            ]
            if backend == ["tvmrt", "tvmllvm"]:
                assert self.tvm_src_dir is not None
                assert self.tvm_src_dir.is_dir()
                to_copy += [
                    (src / "rt_wrapper.c", dest_base),
                    (self.tvm_src_dir / "3rdparty/dlpack/include/dlpack", dest_base / "dlpack"),
                    (
                        self.tvm_src_dir / "include/tvm/runtime/crt/error_codes.h",
                        dest_base / "tvm/runtime/crt/error_codes.h",
                    ),
                    (
                        self.tvm_src_dir / "include/tvm/runtime/crt/logging.h",
                        dest_base / "tvm/runtime/crt/logging.h",
                    ),
                    (
                        self.tvm_src_dir / "include/tvm/runtime/crt/platform.h",
                        dest_base / "tvm/runtime/crt/platform.h",
                    ),
                    (
                        self.tvm_src_dir / "include/tvm/runtime/crt/packed_func.h",
                        dest_base / "tvm/runtime/crt/packed_func.h",
                    ),
                    (
                        self.tvm_src_dir / "include/tvm/runtime/crt/module.h",
                        dest_base / "tvm/runtime/crt/module.h",
                    ),
                    (
                        self.tvm_src_dir / "include/tvm/runtime/crt/func_registry.h",
                        dest_base / "tvm/runtime/crt/func_registry.h",
                    ),
                    (
                        self.tvm_src_dir / "include/tvm/runtime/crt/crt.h",
                        dest_base / "tvm/runtime/crt/crt.h",
                    ),
                    (
                        self.tvm_src_dir / "include/tvm/runtime/crt/graph_executor.h",
                        dest_base / "tvm/runtime/crt/graph_executor.h",
                    ),
                    (
                        self.tvm_src_dir / "include/tvm/runtime/crt/page_allocator.h",
                        dest_base / "tvm/runtime/crt/page_allocator.h",
                    ),
                    (
                        self.tvm_src_dir / "src/runtime/crt/include/tvm/runtime/crt/internal/memory/page_allocator.h",
                        dest_base / "tvm/runtime/crt/internal/memory/page_allocator.h",
                    ),
                    (
                        self.tvm_src_dir
                        / "src/runtime/crt/include/tvm/runtime/crt/internal/graph_executor/graph_executor.h",
                        dest_base / "tvm/runtime/crt/internal/graph_executor/graph_executor.h",
                    ),
                    (
                        self.tvm_src_dir
                        / "src/runtime/crt/include/tvm/runtime/crt/internal/graph_executor/load_json.h",
                        dest_base / "tvm/runtime/crt/internal/graph_executor/load_json.h",
                    ),
                    (
                        self.tvm_src_dir / "src/runtime/crt/include/tvm/runtime/crt/internal/common/ndarray.h",
                        dest_base / "tvm/runtime/crt/internal/common/ndarray.h",
                    ),
                    (
                        self.tvm_src_dir / "src/runtime/crt/graph_executor/graph_executor.c",
                        dest_base / "tvm/runtime/crt/graph_executor/graph_executor.c",
                    ),
                    (
                        self.tvm_src_dir / "src/runtime/crt/graph_executor/load_json.c",
                        dest_base / "tvm/runtime/crt/graph_executor/load_json.c",
                    ),
                    (
                        self.tvm_src_dir / "include/tvm/runtime/c_runtime_api.h",
                        dest_base / "tvm/runtime/c_runtime_api.h",
                    ),
                    (
                        self.tvm_src_dir / "include/tvm/runtime/c_backend_api.h",
                        dest_base / "tvm/runtime/c_backend_api.h",
                    ),
                    (
                        self.tvm_src_dir / "include/tvm/runtime/metadata_types.h",
                        dest_base / "tvm/runtime/metadata_types.h",
                    ),
                    (
                        self.tvm_src_dir / "include/tvm/runtime/crt/stack_allocator.h",
                        dest_base / "tvm/runtime/crt/stack_allocator.h",
                    ),
                    (
                        self.tvm_src_dir / "src/runtime/crt/common/func_registry.c",
                        dest_base / "tvm/runtime/crt/common/func_registry.c",
                    ),
                    (
                        self.tvm_src_dir / "src/runtime/crt/common/crt_runtime_api.c",
                        dest_base / "tvm/runtime/crt/common/crt_runtime_api.c",
                    ),
                    (
                        self.tvm_src_dir / "src/runtime/crt/common/ndarray.c",
                        dest_base / "tvm/runtime/crt/common/ndarray.c",
                    ),
                    (
                        self.tvm_src_dir / "src/runtime/crt/common/packed_func.c",
                        dest_base / "tvm/runtime/crt/common/packed_func.c",
                    ),
                    (
                        self.tvm_src_dir / "src/runtime/crt/common/crt_backend_api.c",
                        dest_base / "tvm/runtime/crt/common/crt_backend_api.c",
                    ),
                    (
                        self.tvm_src_dir / "src/runtime/crt/memory/stack_allocator.c",
                        dest_base / "tvm/runtime/crt/memory/stack_allocator.c",
                    ),
                    (
                        self.tvm_src_dir / "src/runtime/crt/memory/page_allocator.c",
                        dest_base / "tvm/runtime/crt/memory/page_allocator.c",
                    ),
                ]
            elif backend == ["tvmaot", "tvmaotplus"]:
                to_copy += [
                    (src / "aot_wrapper.c", dest_base),
                    (src / "codegen" / "host" / "include", dest_base),
                    (src / "runtime" / "include" / "dlpack", dest_base / "dlpack"),
                    (
                        src / "runtime/include/tvm/runtime/crt/error_codes.h",
                        dest_base / "tvm/runtime/crt/error_codes.h",
                    ),
                    (
                        src / "runtime/include/tvm/runtime/crt/logging.h",
                        dest_base / "tvm/runtime/crt/logging.h",
                    ),
                    (
                        src / "runtime/include/tvm/runtime/crt/platform.h",
                        dest_base / "tvm/runtime/crt/platform.h",
                    ),
                    (
                        src / "runtime/include/tvm/runtime/crt/page_allocator.h",
                        dest_base / "tvm/runtime/crt/page_allocator.h",
                    ),
                    (
                        src / "runtime/include/tvm/runtime/crt/internal/memory/page_allocator.h",
                        dest_base / "tvm/runtime/crt/internal/memory/page_allocator.h",
                    ),
                    (
                        src / "runtime/include/tvm/runtime/c_runtime_api.h",
                        dest_base / "tvm/runtime/c_runtime_api.h",
                    ),
                    (
                        src / "runtime/include/tvm/runtime/c_backend_api.h",
                        dest_base / "tvm/runtime/c_backend_api.h",
                    ),
                    (
                        src / "runtime/include/tvm/runtime/metadata_types.h",
                        dest_base / "tvm/runtime/metadata_types.h",
                    ),
                    (
                        src / "runtime/include/tvm/runtime/crt/stack_allocator.h",
                        dest_base / "tvm/runtime/crt/stack_allocator.h",
                    ),
                    (
                        src / "runtime/src/runtime/crt/common/crt_backend_api.c",
                        dest_base / "tvm/runtime/crt/common/crt_backend_api.c",
                    ),
                    (
                        src / "runtime/src/runtime/crt/memory/stack_allocator.c",
                        dest_base / "tvm/runtime/crt/memory/stack_allocator.c",
                    ),
                    (
                        src / "runtime/src/runtime/crt/memory/page_allocator.c",
                        dest_base / "tvm/runtime/crt/memory/page_allocator.c",
                    ),
                ]
        elif backend in ["tflmi"]:
            to_copy = [
                (src / "model.cc", dest_base),
                (src / "model.cc.h", dest_base),
                (self.mlif_src_dir / "lib/ml_interface/v1/default_data.c", dest_base),
                (self.mlif_src_dir / "lib/ml_interface/v1/ml_interface.h", dest_base),
                (self.mlif_src_dir / "lib/ml_interface/v1/ml_interface.c", dest_base),
                (self.mlif_src_dir / "lib/ml_interface/v1/ml_interface_tflm.cc", dest_base),
                (self.mlif_src_dir / "lib/ml_interface/v1/default_model_support/process_input.c", dest_base),
                (self.mlif_src_dir / "lib/ml_interface/v1/default_model_support/process_output.c", dest_base),
            ]
        else:
            raise ValueError(f"Unsupported Backend: {backend}")

        # Patch makefile
        lines_to_append = []
        if framework != "tflm":
            makefile_exports["SKIP_TFLM"] = "1"
        for key, val in makefile_exports.items():
            if val is None:
                lines_to_append.append(f"export {key}")
            else:
                lines_to_append.append(f"export {key}={val}")
        for key, val in makefile_defines.items():
            if val is None:
                lines_to_append.append(f"DEFINES += {key}")
            else:
                lines_to_append.append(f"DEFINES += {key}={val}")
        for inc in makefile_includes:
            lines_to_append.append(f"INCLUDES += {inc}")
        print("lines_to_append", lines_to_append)
        input("!")
        makefile_path = self.project_dir / "Makefile"
        assert makefile_path.is_file()
        with open(makefile_path, "r") as f:
            makefile_lines = f.read().splitlines()
        makefile_lines = makefile_lines[:-2] + lines_to_append + [makefile_lines[-1]]
        with open(makefile_path, "w") as f:
            f.write("\n".join(makefile_lines))

        # print("to_copy", to_copy)
        for file_or_dir, dest in to_copy:
            # print("file_or_dir")
            assert file_or_dir.exists(), f"Missing file or dir: {file_or_dir}"
            if file_or_dir.is_dir():
                shutil.copytree(file_or_dir, dest, dirs_exist_ok=True)
            else:
                if dest.is_dir():
                    dest_ = dest / file_or_dir.name
                else:
                    dest_ = dest
                dest.parent.mkdir(exist_ok=True, parents=True)
                shutil.copyfile(file_or_dir, dest_)
        # TODO: call make (without renode)?
        out = ""

        return out

    def compile(self, target, src=None):
        # TODO: call make (without renode)?
        out = self.prepare(target, src=src)
        out += utils.make(
            "software", cwd=self.project_dir, env=self.prepare_environment(target=target), live=self.print_outputs
        )
        return out

    def generate(self, src, target, model=None):
        # TODO: implement
        artifacts = []
        out = self.compile(target, src=src)
        elf_name = "software.elf"
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
            "cfu_playground_out.log", content=out, fmt=ArtifactFormat.TEXT  # TODO: split into one file per command
        )  # TODO: rename to tvmaot_out.log?
        artifacts.append(stdout_artifact)
        return {"default": artifacts}, {"default": metrics}

    def flash(self, elf, target, timeout=120):
        # Ignore elf, as we use self.project_dir instead
        # TODO: add alternative approach which allows passing elf instead
        if elf is not None:
            logger.debug("Ignoring ELF file for cfu platform")
        # TODO: implement timeout
        # TODO: make sure that already compiled? -> error or just call compile routine?
        if self.use_renode:
            pass
        else:
            raise NotImplementedError("Only renode is supported")
            if self.wait_for_user:  # INTERACTIVE
                answer = input(
                    f"Make sure that the device '{target.name}' is connected before you press [Enter]"
                    + " (Type 'Abort' to cancel)"
                )
                if answer.lower() == "abort":
                    return ""
            logger.debug("Flashing target software")

    def monitor(self, target, timeout=60):
        # if self.flash_only:
        #     return ""
        # TODO: make renode or FPGA?
        if self.use_renode:
            pass
        else:
            raise NotImplementedError("Only renode is supported")
        out = ""
        out += utils.make(
            "renode-test",
            'TEST_FLAGS="--show-log"',
            cwd=self.project_dir,
            env=self.prepare_environment(target=target),
            live=self.print_outputs,
        )
        return out
