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
import time
import shutil
import tempfile
from pathlib import Path
import pkg_resources


from mlonmcu.setup import utils
from mlonmcu.artifact import Artifact, ArtifactFormat
from mlonmcu.target.metrics import Metrics
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
        # "device": "digilent_arty",  # TODO: FPGA support?
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
        # value2 = self.config["mlif.template"]
        if value is None:
            return None
        return Path(value)

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
        ret = get_cfu_playground_platform_targets()
        return ret

    def create_target(self, name):
        supported = self.get_supported_targets()
        assert name in supported, f"{name} is not a valid CFU Playground device"
        base = supported[name]
        return create_cfu_playground_platform_target(name, self, base=base)

    @property
    def out_dir(self):
        return self.project_dir / "out"

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
        # if val is None:
        if val is None:
            # val = "3"
            return None
        else:
            val = str(val)
        assert val in ["0", "1", "2", "3", "s"], f"Unsupported: {val}"
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

        # print("defines", defines)
        return defines

    def get_makefile_exports(self):
        exports = self.definitions
        exports["TEMPLATE"] = self.mlif_template
        # if self.template_version:
        #     definitions["TEMPLATE_VERSION"] = self.mlif_template_version
        # definitions["TOOLCHAIN"] = self.toolchain
        if self.optimize is not None:
            exports["OPTIMIZE"] = self.optimize

        # print("exports", exports)
        return exports

    def get_makefile_includes(self):
        includes = []

        # print("includes", includes)
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
        # print("new_path", new_path)
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
        # print("src", src)
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
            # print("mlif_template_dir", mlif_template_dir)
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
            if backend in ["tvmrt", "tvmllvm"]:
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
            elif backend in ["tvmaot", "tvmaotplus"]:
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
        # print("to_copy", to_copy)
        # input("!!!")

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
        if target.cpu_variant:
            lines_to_append.append(f"export EXTRA_LITEX_ARGS += --cpu-variant {target.cpu_variant}")
        lines_to_append.append(f"export EXTRA_LITEX_ARGS += --workdir {self.project_dir}")
        if self.num_threads:
            lines_to_append.append(f"export BUILD_JOBS={self.num_threads}")
            lines_to_append.append(f"export JOBS={self.num_threads}")
        lines_to_append.append("export LIBC_CLEANUP=1")
        # print("lines_to_append", lines_to_append
        # input("!")
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
            "software",
            f"OUT_DIR={self.out_dir}",
            f"SOC_BUILD_DIR={self.out_dir}",
            *(["PLATFORM=sim"] if target.rtl_sim else []),
            cwd=self.project_dir,
            env=self.prepare_environment(target=target),
            live=self.print_outputs,
            threads=self.num_threads,
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
        if target.use_renode:
            pass
        elif target.rtl_sim:
            # TODO: move to flash to avoid output?
            out = ""
            out_dir = self.out_dir
            env_ = self.prepare_environment(target=target)
            env_["LIBC_CLEANUP"] = "1"
            out += utils.make(
                "load2",
                *(["PLATFORM=sim"] if target.rtl_sim else []),
                f"OUT_DIR={out_dir}",
                f"SOC_BUILD_DIR={out_dir}",
                cwd=self.project_dir,
                env=env_,
                live=self.print_outputs,
                threads=self.num_threads,
            )
            gateware_dir = out_dir / "gateware"
            print("gateware_dir", gateware_dir)
            assert gateware_dir.is_dir()
            out += utils.execute(
                "bash",
                "build_sim.sh",
                cwd=gateware_dir,
                env=self.prepare_environment(target=target),
                live=self.print_outputs,
            )
            vsim = gateware_dir / "obj_dir" / "Vsim"
            assert vsim.is_file()
        else:
            raise NotImplementedError("Only renode & verilator sim is supported (no FPGAs)")
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
        if target.use_renode:
            out = ""
            out += utils.make(
                "renode-test",
                'TEST_FLAGS="--show-log"',
                cwd=self.project_dir,
                env=self.prepare_environment(target=target),
                live=self.print_outputs,
                threads=self.num_threads,
            )
        elif target.rtl_sim:
            out_dir = self.out_dir
            gateware_dir = out_dir / "gateware"
            vsim = gateware_dir / "obj_dir" / "Vsim"
            assert vsim.is_file()
            # stdin_data = b"3"
            # out += utils.execute(
            #     vsim,
            #     cwd=out_dir / "gateware",
            #     env=self.prepare_environment(target=target),
            #     # live=self.print_outputs,
            #     live=False,
            #     stdin_data=stdin_data,
            # )
            import subprocess
            import signal
            import time
            import select
            import fcntl

            def _kill_monitor():
                pass

            def _set_nonblock(fd):
                flag = fcntl.fcntl(fd, fcntl.F_GETFL)
                fcntl.fcntl(fd, fcntl.F_SETFL, flag | os.O_NONBLOCK)
                new_flag = fcntl.fcntl(fd, fcntl.F_GETFL)
                assert (new_flag & os.O_NONBLOCK) != 0, "Cannot set file descriptor {fd} to non-blocking"

            def _await_ready(rlist, wlist, timeout_sec=None, end_time=None):
                if timeout_sec is None and end_time is not None:
                    timeout_sec = max(0, end_time - time.monotonic())

                rlist, wlist, xlist = select.select(rlist, wlist, rlist + wlist, timeout_sec)
                if not rlist and not wlist and not xlist:
                    raise RuntimeError("Timeout?")

                return True

            def _monitor_helper(verbose=False, start_match=None, end_match=None, timeout=60):
                # start_match and end_match are inclusive
                if timeout:
                    pass  # TODO: implement timeout
                found_start = start_match is None
                found_menu = False
                menu_match = "Q: Exit"
                # TODO: log command!
                outStr = ""
                process = subprocess.Popen(
                    vsim,
                    cwd=out_dir / "gateware",
                    env=self.prepare_environment(target=target),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  # TODO: store stdout?
                    stdin=subprocess.PIPE,
                    bufsize=0,
                )
                _set_nonblock(process.stdin.fileno())

                try:
                    exit_code = None
                    for line in process.stdout:
                        new_line = line.decode(errors="replace")
                        if verbose:
                            print(new_line.replace("\n", ""))
                        if not found_menu:
                            if menu_match in new_line:
                                # print("FOUND MENU")
                                found_menu = True
                                # process.stdin.write(b"3\n")
                                data = b"3"
                                fd = process.stdin.fileno()
                                _await_ready([], [fd], end_time=None)
                                _ = os.write(fd, data)
                                # num_written = os.write(fd, data)
                                # print("num_written", num_written)
                        else:
                            if start_match and start_match in new_line:
                                outStr = new_line
                                found_start = True
                            else:
                                outStr = outStr + new_line
                            if found_start:
                                if end_match and end_match in new_line:
                                    # _kill_monitor()
                                    process.terminate()
                                    exit_code = 0
                    while exit_code is None:
                        exit_code = process.poll()
                    if not verbose and exit_code != 0:
                        logger.error(outStr)
                    cmd = "TODO"
                    assert exit_code == 0, "The process returned an non-zero exit code {}! (CMD: `{}`)".format(
                        exit_code, cmd
                    )
                except KeyboardInterrupt:
                    logger.debug("Interrupted subprocess. Sending SIGINT signal...")
                    _kill_monitor()
                    pid = process.pid
                    os.kill(pid, signal.SIGINT)
                # os.system("reset")
                return outStr

            logger.debug("Monitoring verilator")
            # TODO: do not drop verilator stdout/stderr?
            return _monitor_helper(
                verbose=self.print_outputs,
                start_match="Program start.",
                end_match="Program finish.",  # TODO: missing exit code?
                timeout=timeout,
            )
        else:
            raise NotImplementedError("Only renode is supported")
        return out

    def run(self, elf, target, timeout=120):
        # Only allow one serial communication at a time
        # with FileLock(Path(tempfile.gettempdir()) / "mlonmcu_serial.lock"):
        metrics = Metrics()
        start_time = time.time()
        self.flash(elf, target, timeout=timeout)
        end_time = time.time()
        diff = end_time - start_time
        start_time = time.time()
        output = self.monitor(target, timeout=timeout)
        end_time = time.time()
        diff2 = end_time - start_time
        if target.use_renode:
            metrics.add("Renode Build Time [s]", diff, True)
            metrics.add("Renode Monitor Time [s]", diff2, True)
            metrics.add("Simulation Time [s]", diff2, True)
        elif target.rtl_sim:
            metrics.add("Verilator Build Time [s]", diff, True)
            metrics.add("Verilator Monitor Time [s]", diff2, True)
            metrics.add("Simulation Time [s]", diff2, True)
        else:
            metrics.add("FPGA Flash Time [s]", diff, True)
            metrics.add("FPGA Monitor Time [s]", diff2, True)

        return output, metrics
