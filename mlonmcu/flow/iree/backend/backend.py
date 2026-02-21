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
"""Generic IREEBackend implementation."""
import os
import tempfile
from pathlib import Path
from typing import Tuple, Optional
import multiprocessing

from mlonmcu.flow.backend import Backend
from mlonmcu.setup import utils
from mlonmcu.timeout import exec_timeout
from mlonmcu.config import str2bool
from mlonmcu.logging import get_logger
from mlonmcu.target.elf import get_code_size_from_static_lib
from mlonmcu.models.model_info import (
    get_model_info,
    get_supported_formats_iree,
)
from mlonmcu.target.metrics import Metrics
from mlonmcu.artifact import Artifact, ArtifactFormat
from .wrapper import generate_iree_wrapper


logger = get_logger()


def get_iree_compile_hal_backend_target_args(hal_backend, target_details):
    """Get LLVM-CPU specific arguments."""
    if hal_backend != "llvm-cpu":
        return []

    def helper(value):
        if isinstance(value, (bool, int)):
            # value = "true" if value else "false"
            value = str(int(value))
        return value

    return sum(
        [[f"--iree-llvmcpu-target-{key}", helper(value)] for key, value in target_details.items()],
        [],
    )


def get_iree_compile_optimization_args(
    iree_version: Optional[str] = None,
    opt_level: Optional[str] = None,
):
    """Get optimization-related arguments."""
    if iree_version is None:
        logger.warning("iree.version undefined, assuming v3.3")
        major = 3
        minor = 3
    else:
        major, minor = map(int, iree_version.split(".", 1))
    ret = []
    if major < 3 or (major == 3 and minor < 3):
        # No unified optimization flags
        if opt_level is not None:
            raise ValueError("opt_level not supported for IREE version {iree_vection}")
        ret += [
            "--iree-opt-aggressively-propagate-transposes",
            "--iree-dispatch-creation-enable-aggressive-fusion",
            "--iree-input-demote-i64-to-i32=true",
            "--iree-stream-resource-index-bits=8",
        ]
    else:
        if opt_level is not None:
            ret += ("--iree-opt-level={opt_level}",)  # needs iree 3.3+
    # TODO: vm-bytecode only?
    ret += [
        "--iree-vm-bytecode-module-strip-source-map=true",
        "--iree-vm-emit-polyglot-zip",
    ]
    return ret


def get_iree_compile_llvmcpu_vectorization_unroll_args(
    hal_backend: str,
    target_vector_width: Optional[int] = None,
    target_scalable_vector: Optional[bool] = None,
    loop_unroll: Optional[bool] = None,
):
    """Get vectorization-related arguments."""
    if hal_backend != "llvm-cpu":
        return []
    supports_vectorization = target_vector_width is not None and target_vector_width > 0
    vector_width = 0
    if supports_vectorization:
        vector_width = target_vector_width
    else:
        vector_width = 4
    ret = [
        f"--iree-llvmcpu-target-vector-width-in-bytes={vector_width}",
        f"--iree-llvmcpu-slp-vectorization={int(supports_vectorization)}",
        f"--iree-llvmcpu-loop-vectorization={int(supports_vectorization)}",
        f"--iree-llvmcpu-disable-vector-peeling={1-int(supports_vectorization)}",
        # "--iree-llvmcpu-check-linalg-vectorization=0",
        *(
            [f"--iree-llvmcpu-enable-scalable-vectorization={int(target_scalable_vector)}"]
            if target_scalable_vector is not None
            else []
        ),
        f"--iree-llvmcpu-fail-on-large-vector={1-int(supports_vectorization)}",
        *([f"--iree-llvmcpu-loop-unrolling={int(loop_unroll)}"] if loop_unroll is not None else []),
    ]
    return ret


class IREEBackend(Backend):
    """Base IREE Backend."""

    registry = {}

    name = None

    FEATURES = set()

    DEFAULTS = {
        "print_outputs": False,
        "opt_level": None,
        "target_cpu": None,
        "target_triple": None,
        "target_abi": None,
        "target_cpu_features": None,
        "iree_compile_extra_args": [],
        "num_threads": multiprocessing.cpu_count(),
        "strip_assertions": None,
        "target_vector_width": None,
        "target_scalable_vectorization": None,
        "loop_unroll": True,
    }

    OPTIONAL = {"iree.version", "iree.build_dir"}

    REQUIRED = {"iree.install_dir", "iree.src_dir"}

    def __init__(self, output_format=None, hal_backend=None, hal_inline=False, features=None, config=None):
        super().__init__(framework="iree", features=features, config=config)
        self.identifier = "model"
        assert output_format in ["vm-bytecode", "vm-c"]
        self.output_format = output_format
        assert hal_backend in ["vmvx", "llvm-cpu"]
        self.hal_backend = hal_backend
        self.hal_inline = hal_inline
        self.execution_model = None
        self.static_lib = self.hal_backend == "llvm-cpu"
        if self.hal_inline:
            if self.hal_backend == "vmvx":
                self.hal_backend = "vmvx-inline"
                self.execution_model = "inline-static"
            elif self.hal_backend == "llvm-cpu":
                self.execution_model = "inline-dynamic"

        self.model = None  # Actual filename!
        self.model_info = None
        self.input_shapes = None
        self.model_format = None
        self.supported_formats = get_supported_formats_iree()
        # self.supported_formats = [ModelFormats.TFLITE, ModelFormats.MLIR]
        # self.supported_formats = [ModelFormats.MLIR]

        # self.prefix = "default"
        self.artifacts = []

    # @property
    # def target_device(self):
    #     return self.config["target_device"]

    @property
    def target_cpu(self):
        return self.config["target_cpu"]

    @property
    def target_triple(self):
        return self.config["target_triple"]

    @property
    def target_abi(self):
        return self.config["target_abi"]

    @property
    def target_cpu_features(self):
        return self.config["target_cpu_features"]

    @property
    def opt_level(self):
        # O3, O2,...
        return self.config["opt_level"]

    @property
    def iree_compile_extra_args(self):
        return self.config["iree_compile_extra_args"]

    @property
    def iree_install_dir(self):
        return self.config["iree.install_dir"]

    @property
    def iree_build_dir(self):
        ret = self.config["iree.build_dir"]
        if ret is None:
            return None
        return Path(ret)

    @property
    def iree_src_dir(self):
        return self.config["iree.src_dir"]

    @property
    def iree_compile_exe(self):
        return Path(self.iree_install_dir) / "bin" / "iree-compile"

    @property
    def iree_c_embed_data_exe(self):
        return Path(self.iree_install_dir) / "bin" / "iree-c-embed-data"

    @property
    def mlir_opt_exe(self):
        iree_build_dir = self.iree_build_dir
        assert iree_build_dir is not None, "Missing: iree.build_dir"
        return self.iree_build_dir / "llvm-project" / "bin", "mlir-opt"

    @property
    def iree_tflite_path(self):
        return Path(self.iree_src_dir) / "integrations" / "tensorflow" / "python_projects" / "iree_tflite"

    @property
    def iree_tf_path(self):
        return Path(self.iree_src_dir) / "integrations" / "tensorflow" / "python_projects" / "iree_tf"

    @property
    def print_outputs(self):
        value = self.config["print_outputs"]
        return str2bool(value)

    @property
    def num_threads(self):
        return self.config["num_threads"]

    def prepare_environment(self):
        env = os.environ.copy()
        pythonpath = env.get("PYTHONPATH", "")
        pythonpath = f"{self.iree_tflite_path}:{self.iree_tf_path}:{pythonpath}"
        env["PYTHONPATH"] = pythonpath
        old_path = env.get("PATH", "")
        new_path = f"{self.iree_install_dir}/bin:{old_path}"
        env["PATH"] = new_path
        return env

    def get_target_details(self):
        ret = {}
        if self.target_cpu:
            ret["cpu"] = self.target_cpu
        if self.target_triple:
            ret["triple"] = self.target_triple
        if self.target_abi:
            ret["abi"] = self.target_abi
        if self.target_cpu_features:
            temp = self.target_cpu_features
            # custom_unroll = False  # TODO
            custom_unroll = True
            if custom_unroll:
                temp += ",+no-default-unroll"
            ret["cpu-features"] = temp
        return ret

    def get_iree_c_embed_data_args(self, vmfb_in, impl_out, header_out):
        return [
            vmfb_in,
            f"--output_header={header_out}",
            f"--output_impl={impl_out}",
            f"--identifier={self.identifier}",
            "--flatten",
        ]

    @property
    def strip_assertions(self):
        return str2bool(self.config["strip_assertions"], allow_none=True)

    @property
    def iree_version(self):
        value = self.config["iree.version"]
        if value is None:
            return None
        if not isinstance(value, str):
            value = str(float(value))
        assert value.count(".") == 1
        return value

    @property
    def target_vector_width(self):
        return self.config["target_vector_width"]

    @property
    def target_scalable_vector(self):
        return str2bool(self.config["target_scalable_vectorization"], allow_none=True)

    @property
    def loop_unroll(self):
        return str2bool(self.config["loop_unroll"], allow_none=True)

    def get_iree_compile_args(self, out, model_path):
        static_lib_path = out.parent / f"{self.identifier}_static_lib.o"
        args = [
            model_path,
            # TODO: use true/false
            *(
                [f"--iree-opt-strip-assertions={int(self.strip_assertions)}"]
                if self.strip_assertions is not None
                else []
            ),
            *get_iree_compile_llvmcpu_vectorization_unroll_args(
                self.hal_backend, self.target_vector_width, self.target_scalable_vector, self.loop_unroll
            ),
            *get_iree_compile_optimization_args(self.iree_version, self.opt_level),
            f"--output-format={self.output_format}",
            f"--iree-hal-target-backends={self.hal_backend}",
            *get_iree_compile_hal_backend_target_args(self.hal_backend, self.get_target_details()),
            *([f"--iree-execution-model={self.execution_model}"] if self.execution_model is not None else []),
            # TODO: emitc only?
            *(
                [
                    "--iree-hal-target-device=local",
                    "--iree-hal-local-target-device-backends=llvm-cpu",
                    "--iree-vm-target-index-bits=32",
                    "--iree-llvmcpu-link-static",
                    "--iree-llvmcpu-link-embedded=false",
                    f"-iree-llvmcpu-static-library-output-path={static_lib_path}",
                ]
                # if self.output_format == "vm-c" and self.hal_backend == "llvm-cpu"
                if self.hal_backend == "llvm-cpu" and self.static_lib
                else []
            ),
            *(
                [
                    "--iree-llvmcpu-debug-symbols=false",  # TODO: expose
                ]
                if self.hal_backend == "llvm-cpu"
                else []
            ),
            # "--iree-stream-partitioning-favor=min-peak-memory",  # TODO: expose & check
            # *get_target_tvmc_args(
            #     self.target,
            #     extra_targets=self.extra_targets,
            #     target_details=self.get_target_details(),
            #     extra_target_details=self.extra_target_details,
            # ),
            # *get_runtime_executor_tvmc_args(self.runtime, self.executor),
            # *get_pass_config_tvmc_args(self.pass_config),
            # *get_disabled_pass_tvmc_args(self.disabled_passes),
            # *get_input_shapes_tvmc_args(self.input_shapes),
            # *get_tuning_records_tvmc_args(self.use_tuning_results, self.get_tuning_records()),
            # *get_desired_layout_args(self.desired_layout, self.desired_layout_ops, self.desired_layout_map),
            # *(["--dump-code", ",".join(dump)] if dump is not None and len(dump) > 0 else []),
            *self.iree_compile_extra_args,
            # *["--opt-level", str(self.opt_level)],
            *["-o", str(out)],
            # *["-f", self.fmt],
            # *["--model-format", self.model_format],
        ]
        return args

    def invoke_iree(self, exe, *args, cwd=None, **kwargs):
        return utils.execute(exe, *args, live=self.print_outputs, cwd=cwd, **kwargs)

    def invoke_iree_compile(self, out, model_path, cwd=None):
        args = self.get_iree_compile_args(out, model_path)
        self.timeout_sec = 0
        if self.timeout_sec > 0:
            ret = exec_timeout(
                self.timeout_sec,
                self.invoke_iree,
                self.iree_compile_exe,
                *args,
                cwd=cwd,
            )
        else:
            ret = self.invoke_iree(self.iree_compile_exe, *args, cwd=cwd)
        return ret

    def translate_mlirbc_to_mlir(self, mlirbc_path, mlir_path, cwd=None):
        args = [
            mlirbc_path,
            "-o",
            mlir_path,
            "--compile-to",
            "input",
        ]
        self.timeout_sec = 0
        if self.timeout_sec > 0:
            ret = exec_timeout(
                self.timeout_sec,
                self.invoke_iree,
                self.iree_compile_exe,
                *args,
                cwd=cwd,
            )
        else:
            ret = self.invoke_iree(self.iree_compile_exe, *args, cwd=cwd)
        return ret

    def invoke_iree_c_embed_data(self, vmfb_file, impl_file, header_file, cwd=None):
        args = self.get_iree_c_embed_data_args(vmfb_file, impl_file, header_file)
        self.timeout_sec = 0
        if self.timeout_sec > 0:
            ret = exec_timeout(
                self.timeout_sec,
                self.invoke_iree,
                self.iree_c_embed_data_exe,
                *args,
                cwd=cwd,
            )
        else:
            ret = self.invoke_iree(self.iree_c_embed_data_exe, *args, cwd=cwd)
        return ret

    def load_model(
        self, model, input_shapes=None, output_shapes=None, input_types=None, output_types=None, params_path=None
    ):
        assert params_path is None
        self.model = model
        self.model_format, self.model_info = get_model_info(model, backend_name=self.name)
        # TODO: path model class instead of path!
        # fmt = self.model.formats[0]
        # need_model_info = True
        # if input_shapes:
        #     self.input_shapes = input_shapes
        #     if output_shapes and input_types and output_types:
        #         need_model_info = False
        #         self.model_format, self.model_info = get_fallback_model_info(
        #             model, input_shapes, output_shapes, input_types, output_types, backend_name=self.name
        #         )
        # else:
        #     self.input_shapes = None  # Relevant for multiple subs using the same backend
        # if need_model_info:
        #     try:
        #         self.model_format, self.model_info = get_model_info(model, backend_name=self.name)
        #     except Exception as e:
        #         self.model_format = get_model_format(model)
        #         if self.model_format != "relay":
        #             logger.warning(
        #                 "Fetching of Model Info failed (%s). Falling back to Relay-based info.", type(e).__name__
        #             )
        #             self.model_info = None
        #         else:
        #             raise e

        #     if self.model_info:
        #         # TODO: also handle output_shapes
        #         # TODO: take care of refresh_model_info
        #         if self.input_shapes:
        #             self.model_info.in_tensors = [t for t in self.model_info.in_tensors
        #                                               if t.name in self.input_shapes]
        #             assert (
        #                 len(self.model_info.in_tensors) > 0
        #             ), "Missmatch between provided input names and detected ones"
        #         else:
        #             self.input_shapes = {tensor.name: tensor.shape for tensor in self.model_info.in_tensors}
        # if self.model_info:
        #     self.model_info.validate()

    @property
    def use_emitc(self):
        return self.output_format == "vm-c"

    def generate(self) -> Tuple[dict, dict]:
        artifacts = []
        metrics = Metrics()
        assert self.model is not None
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            # out_dir = Path("/tmp/iree_out/")
            model_path = self.model
            model_info = self.model_info
            translated = False
            mlir_path = out_dir / f"{self.identifier}.mlir"
            if self.model_format == "mlir":
                # Copy model to model.mlir for consistent function_names
                utils.copy(model_path, mlir_path)
                translated = True
            else:
                translated = True
                mlirbc_path = out_dir / f"{self.identifier}.mlirbc"
                needs_mlirbc2mlir = False
                if self.model_format == "tflite":
                    iree_version = self.iree_version
                    # TODO: move to utils
                    if iree_version is None:
                        logger.warning("iree.version undefined, assuming v3.3")
                        major = 3
                        minor = 3
                    else:
                        major, minor = map(int, iree_version.split(".", 1))
                    if major == 3 and minor <= 1:
                        python_args = [
                            "-m",
                            "iree.tools.tflite.scripts.iree_import_tflite",
                            model_path,
                            "-o",
                            mlirbc_path,
                        ]
                        utils.python(
                            *python_args, live=self.print_outputs, env=self.prepare_environment(), cwd=temp_dir
                        )
                        needs_mlirbc2mlir = True
                    elif major == 3 and minor > 5:  # TODO: check
                        tosa_converter_args = ["tosa-converter-for-tflite", model_path, "--text", "-o", mlir_path]
                        utils.execute(
                            *tosa_converter_args, live=self.print_outputs, env=self.prepare_environment(), cwd=temp_dir
                        )
                        fix_mlir = True
                        if fix_mlir:
                            attach_tosa_target = True
                            fix_dynamic_shapes = True
                            temp_mlir_path = out_dir / "temp.mlir"
                            utils.copy(mlir_path, temp_mlir_path)
                            mlir_opt_exe = self.mlir_opt_exe
                            assert mlir_opt_exe is not None, "Undefined: mlir_opt_exe"
                            assert mlir_opt_exe.is_file(), f"Missing file: {mlir_opt_exe}"
                            mlir_opt_args = [
                                mlir_opt_exe,
                                temp_mlir_path,
                                "-o",
                                mlir_path,
                            ]
                            if fix_dynamic_shapes:
                                args_temp = {}
                                for i, in_tensor in enumerate(model_info.in_tensors):
                                    arg_name = f"arg{i}"
                                    input_shape = in_tensor.shape
                                    arg_shape_str = "x".join(input_shape)
                                    args_temp[arg_name] = arg_shape_str
                                args_str = ",".join([f"arg{i}:1x1960" for arg_name, arg_shape_str in args_temp.items()])
                                args_str_ = f"args={args_str}"
                                mlir_opt_args.append(f'--tosa-experimental-input-shape="{args_str_}"')
                                mlir_opt_args.append("-tosa-infer-shapes")
                            if attach_tosa_target:  # TODO: not working because this is hardcoded in IREE (see TODO)
                                tosa_target_str = "specification_version=1.1.draft profiles=pro_int,pro_fp extensions=int16,int4,int64,bf16,fp8e4m3,fp8e5m2,fft,variable,controlflow,doubleround,inexactround,mxfp_conv,shape"
                                mlir_opt_args.append(f"-tosa-attach-target={tosa_target_str}")
                    elif major == 3 and minor > 1:
                        raise RuntimeError("TFLite (TOSA) importer unsupported for iree.version <3.1")
                    elif major < 3:
                        raise RuntimeError("IREE version < 3.0 is untested/unsupported.")
                elif self.model_format == "onnx":
                    # opset_version = 17
                    opset_version = None
                    python_args = [
                        "-m",
                        # "iree.tools.onnx.scripts.iree_import_onnx",
                        "iree.compiler.tools.import_onnx",
                        model_path,
                        "-o",
                        mlir_path,
                        *([f"--opset-version={opset_version}"] if opset_version is not None else []),
                    ]
                    utils.python(*python_args, live=self.print_outputs, env=self.prepare_environment(), cwd=temp_dir)
                elif self.model_format in ["saved_model", "pb"]:
                    python_args = [
                        "-m",
                        "iree.tools.tf.scripts.iree_import_tf",
                        model_path,
                        "-o",
                        mlir_path,
                        "--tf-import-type=savedmodel_v1",
                        "--tf-savedmodel-exported-names=predict",
                    ]
                    utils.python(*python_args, live=self.print_outputs, env=self.prepare_environment(), cwd=temp_dir)
                else:
                    raise NotImplementedError(f"Unhandled format: {self.model_format}")
                if needs_mlirbc2mlir:
                    self.translate_mlirbc_to_mlir(mlirbc_path, mlir_path, cwd=temp_dir)
                model_format, model_info = get_model_info(mlir_path, backend_name=self.name)
                assert model_format == "mlir"
                with open(mlir_path, "r") as f:
                    mlir_content = f.read()
                artifacts.append(
                    Artifact(
                        mlir_path.name,
                        content=mlir_content,
                        fmt=ArtifactFormat.SOURCE,
                    )
                )
                if needs_mlirbc2mlir:
                    with open(mlirbc_path, "rb") as f:
                        mlirbc_raw = f.read()
                    artifacts.append(
                        Artifact(
                            mlirbc_path.name,
                            raw=mlirbc_raw,
                            fmt=ArtifactFormat.BIN,
                        )
                    )
                # model_path = mlirbc_path
            model_path = mlir_path
            if self.output_format == "vm-bytecode":
                out_path = out_dir / f"{self.identifier}.vmfb"
            elif self.output_format == "vm-c":
                out_path = out_dir / f"{self.identifier}_emitc.h"
            out = self.invoke_iree_compile(out_path, model_path, cwd=temp_dir)
            if self.hal_backend == "llvm-cpu":
                static_lib_path = out_dir / f"{self.identifier}_static_lib.o"
                header_path = out_dir / f"{self.identifier}_static_lib.h"
                with open(static_lib_path, "rb") as f:
                    static_lib_raw = f.read()
                with open(header_path, "r") as f:
                    header_content = f.read()
                kernels_code_size = get_code_size_from_static_lib(static_lib_path)
                metrics.add("Kernels Size", kernels_code_size, True)

                artifacts.append(
                    Artifact(
                        static_lib_path.name,
                        raw=static_lib_raw,
                        fmt=ArtifactFormat.BIN,
                    )
                )
                artifacts.append(
                    Artifact(
                        header_path.name,
                        content=header_content,
                        fmt=ArtifactFormat.SOURCE,
                    )
                )

                if self.output_format == "vm-c":
                    with open(out_path, "r") as f:
                        emitc_content = f.read()

                    artifacts.append(
                        Artifact(
                            out_path.name,
                            content=emitc_content,
                            fmt=ArtifactFormat.SOURCE,
                        )
                    )
            # elif self.output_format == "vm-bytecode":
            # elif self.hal_backend in ["vmvx", "vmvx-inline"]:
            #     assert self.output_format == "vm-bytecode"
            if self.output_format == "vm-bytecode":
                with open(out_path, "rb") as f:
                    out_raw = f.read()
                artifacts.append(
                    Artifact(
                        out_path.name,
                        raw=out_raw,
                        fmt=ArtifactFormat.BIN,
                    )
                )
                impl_path = out_dir / f"{self.identifier}.c"
                header_path = out_dir / f"{self.identifier}.h"
                out += self.invoke_iree_c_embed_data(out_path, impl_path, header_path, cwd=temp_dir)
                with open(impl_path, "r") as f:
                    impl_content = f.read()
                with open(header_path, "r") as f:
                    header_content = f.read()
                artifacts.append(
                    Artifact(
                        impl_path.name,
                        content=impl_content,
                        fmt=ArtifactFormat.SOURCE,
                    )
                )
                artifacts.append(
                    Artifact(
                        header_path.name,
                        content=header_content,
                        fmt=ArtifactFormat.SOURCE,
                    )
                )
            wrapper_content, wrapper_header_content, sync_content, utils_content = generate_iree_wrapper(
                model_info,
                self.identifier,
                use_emitc=self.use_emitc,
                vmvx=self.hal_backend in ["vmvx", "vmvx-inline"],
                translated=translated,
            )
            artifacts.append(
                Artifact(
                    "iree_wrapper.c",
                    content=wrapper_content,
                    fmt=ArtifactFormat.SOURCE,
                )
            )
            artifacts.append(
                Artifact(
                    f"{self.identifier}_utils.c",
                    content=utils_content,
                    fmt=ArtifactFormat.SOURCE,
                )
            )
            artifacts.append(
                Artifact(
                    "iree_wrapper.h",
                    content=wrapper_header_content,
                    fmt=ArtifactFormat.SOURCE,
                )
            )
            # if not self.use_emitc:
            if True:
                artifacts.append(
                    Artifact(
                        "device_sync.c",
                        content=sync_content,
                        fmt=ArtifactFormat.SOURCE,
                    )
                )
            # artifacts.append(
            #     Artifact(
            #         f"{self.prefix}.params",
            #         raw=params,
            #         fmt=ArtifactFormat.RAW,
            #     )
            # )
            stdout_artifact = Artifact(
                "iree_compile_out.log", content=out, fmt=ArtifactFormat.TEXT
            )  # TODO: rename to tvmaot_out.log?
            artifacts.append(stdout_artifact)
        print("artifacts", artifacts)
        return {"default": artifacts}, {"default": metrics}

    def get_platform_defs(self, platform):
        ret = super().get_platform_defs(platform)
        ret["IREE_EMITC"] = self.use_emitc
        if self.hal_inline and self.hal_backend == "llvm-cpu":
            ret["IREE_LOADER_HAL"] = True
        elif self.hal_backend == "vmvx":
            ret["IREE_VMVX"] = True
        elif self.hal_backend == "vmvx-inline":
            ret["IREE_INLINE_HAL"] = True
        return ret
