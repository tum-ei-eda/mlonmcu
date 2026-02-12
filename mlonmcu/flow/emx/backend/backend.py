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
import os
import string
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
    # get_fallback_model_info,
    # get_supported_formats,
    # get_supported_formats_emx,
    # get_model_format,
)
from mlonmcu.flow.tvm.backend.wrapper import getSizes
from mlonmcu.target.metrics import Metrics
from mlonmcu.artifact import Artifact, ArtifactFormat

from mlonmcu.models.model import ModelFormats

# from .python_utils import prepare_python_environment
# from .tvmc_utils import (
#     get_target_tvmc_args,
#     get_pass_config_tvmc_args,
#     get_disabled_pass_tvmc_args,
#     get_runtime_executor_tvmc_args,
#     get_input_shapes_tvmc_args,
#     get_tuning_records_tvmc_args,
#     get_desired_layout_args,
# )

logger = get_logger()


def fill(template, **kwargs):
    return string.Template(template).substitute(**kwargs)


def generate_emx_wrapper(
    model_info, identifier: str,
):
    def generate_header(prefix="model"):
        upper_prefix = prefix.upper()
        code = f"""
// This file is generated. Do not edit.
#ifndef {upper_prefix}_GEN_H
#define {upper_prefix}_GEN_H

#include <stddef.h>

int {prefix}_init();
void *{prefix}_input_ptr(int index);
size_t {prefix}_input_size(int index);
size_t {prefix}_inputs();
int {prefix}_invoke();
void *{prefix}_output_ptr(int index);
size_t {prefix}_output_size(int index);
size_t {prefix}_outputs();

#endif  // {upper_prefix}_GEN_H
"""
        return code

    def generate_wrapper(prefix="model"):
        out = ""
        def writeTensors(in_tensors, out_tensors):
            retStr = """
"""

            def writeTensorsHelper(tensors, out=False):
                lenTensors = len(tensors)
                direction = "out" if out else "in"
                ret = ""
                names = [f"{direction}put{i}_data" for i in range(lenTensors)]
                for i, t in enumerate(tensors):
                    ret += "char " + names[i] + "[" + str(t.size) + "];\n"
                ret += f"void* {direction}puts[] = {{" + ", ".join(names) + "};\n"
                return ret

            retStr += writeTensorsHelper(in_tensors, False)
            retStr += writeTensorsHelper(out_tensors, True)
            return retStr

        out = ""
        out += '#include "emx_wrapper.h"\n'
        out += "_Bool model_load(const char *path);\n"

        in_arg_types = [f"const {inp.c_type}*" for inp in model_info.in_tensors]
        out_arg_types = [f"{outp.c_type}*" for outp in model_info.out_tensors]
        temp = ", ".join(in_arg_types + out_arg_types)
        out += f"void model({temp});\n"
        out += "\n"
        out += writeTensors(model_info.in_tensors, model_info.out_tensors)

        mainCode = """
int ${prefix}_init()
{
    if (!model_load("model.bin")) {
        return 1;
    }
    return 0;
}

void *${prefix}_input_ptr(int index)
{
    return inputs[index];
}

size_t ${prefix}_input_size(int index)
{
    ${inSizes}
    return sizes[index];
}

size_t ${prefix}_inputs()
{
    return ${numInputs};
}

int ${prefix}_invoke()
{
    model(${model_input_args});
    return 0;  // TODO
}

void *${prefix}_output_ptr(int index)
{
    return outputs[index];
}

size_t ${prefix}_output_size(int index)
{
    ${outSizes}
    return sizes[index];
}

size_t ${prefix}_outputs()
{
    return ${numOutputs};
}
"""
        model_input_args = ", ".join([f"inputs[{idx}]" for idx in range(len(model_info.in_tensors))] + [f"outputs[{idx}]" for idx in range(len(model_info.out_tensors))])
        out += fill(
            mainCode,
            inSizes=getSizes(model_info.in_tensors),
            outSizes=getSizes(model_info.out_tensors),
            numInputs=len(model_info.in_tensors),
            numOutputs=len(model_info.out_tensors),
            prefix=prefix,
            model_input_args=model_input_args,
        )
        return out
    wrapper = generate_wrapper()
    header = generate_header()
    return wrapper, header


class EMXBackend(Backend):
    registry = {}

    # name = None
    name = "emx"

    DEFAULTS = {
        "print_outputs": False,
        "emx_compile_extra_args": [],
    }

    REQUIRED = {"emx.src_dir"}

    def __init__(self, output_format=None, hal_backend=None, hal_inline=False, features=None, config=None):
        super().__init__(framework="emx", features=features, config=config)
        self.identifier = "model"

        self.model = None  # Actual filename!
        self.model_info = None
        self.input_shapes = None
        self.model_format = None
        # self.supported_formats = get_supported_formats_emx()
        # self.supported_formats = [ModelFormats.TFLITE, ModelFormats.MLIR]
        self.supported_formats = [ModelFormats.ONNX]

        self.artifacts = []

    @property
    def emx_compile_extra_args(self):
        return self.config["emx_compile_extra_args"]

    @property
    def emx_src_dir(self):
        return self.config["emx.src_dir"]

    @property
    def emx_compile_extra_args(self):
        return self.config["emx_compile_extra_args"]

    @property
    def print_outputs(self):
        value = self.config["print_outputs"]
        return str2bool(value)

    def prepare_environment(self):
        env = os.environ.copy()
        pythonpath = env.get("PYTHONPATH", "")
        pythonpath = f"{self.emx_src_dir}:{pythonpath}"
        # print("pythonpath", pythonpath)
        env["PYTHONPATH"] = pythonpath
        return env

    def get_emx_compile_args(self, out, model_path):
        args = [
            "compile",
            model_path,
            *self.emx_compile_extra_args,
            str(out),
            # "--verbose",  # TODO: expose
            # f"--model-name={self.identifier}",
            # --emit-data-file, --truncate-weights-after, --large-temp-threshold, --large-weight-threshold, --restrict-arrays, --no-restrict-arrays, --fp32-accumulation-strategy, --fp16-accumulation-strategy
        ]
        return args

    def invoke_emx(self, exe, *args, cwd=None, **kwargs):
        return utils.execute(exe, *args, live=self.print_outputs, cwd=cwd, **kwargs)

    def invoke_emx_compile(self, out, model_path, cwd=None):
        fmt2exe = {
            # ModelFormats.ONNX: "emx-onnx-cgen",
            "onnx": "emx-onnx-cgen",
        }
        exe = fmt2exe.get(self.model_format)
        print("exe", exe)
        assert exe is not None, f"Unsupported format: {self.model_format}"
        args = self.get_emx_compile_args(out, model_path)
        self.timeout_sec = 0
        if self.timeout_sec > 0:
            ret = exec_timeout(
                self.timeout_sec,
                self.invoke_emx,
                exe,
                *args,
                cwd=cwd,
            )
        else:
            ret = self.invoke_emx(exe, *args, cwd=cwd)
        return ret

    def load_model(
        self, model, input_shapes=None, output_shapes=None, input_types=None, output_types=None, params_path=None
    ):
        assert params_path is None
        self.model = model
        self.model_format, self.model_info = get_model_info(model, backend_name=self.name)

    def generate(self) -> Tuple[dict, dict]:
        artifacts = []
        metrics = Metrics()
        assert self.model is not None
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            model_path = self.model
            model_info = self.model_info
            out_file = out_dir / f"{self.identifier}.c"
            out = self.invoke_emx_compile(out_file, model_path, cwd=temp_dir)
            wrapper_content, wrapper_header_content = generate_emx_wrapper(
                model_info,
                self.identifier,
            )
            with open(out_file, "r") as f:
                model_src = f.read()
            artifacts.append(
                Artifact(
                    out_file.name,
                    content=model_src,
                    fmt=ArtifactFormat.SOURCE,
                )
            )
            artifacts.append(
                Artifact(
                    "emx_wrapper.c",
                    content=wrapper_content,
                    fmt=ArtifactFormat.SOURCE,
                )
            )
            artifacts.append(
                Artifact(
                    "emx_wrapper.h",
                    content=wrapper_header_content,
                    fmt=ArtifactFormat.SOURCE,
                )
            )
            stdout_artifact = Artifact(
                "emx_compile_out.log", content=out, fmt=ArtifactFormat.TEXT
            )
            artifacts.append(stdout_artifact)
        print("artifacts", artifacts)
        return {"default": artifacts}, {"default": metrics}

    def get_platform_defs(self, platform):
        ret = super().get_platform_defs(platform)
        return ret
