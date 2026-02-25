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
)
from mlonmcu.flow.tvm.backend.wrapper import getSizes
from mlonmcu.target.metrics import Metrics
from mlonmcu.artifact import Artifact, ArtifactFormat

from mlonmcu.models.model import ModelFormats


logger = get_logger()


def fill(template, **kwargs):
    return string.Template(template).substitute(**kwargs)


def generate_executorch_wrapper(
    model_info, identifier: str,
):
    def generate_header(prefix="model"):
        upper_prefix = prefix.upper()
        code = f"""
// This file is generated. Do not edit.
#ifndef {upper_prefix}_GEN_H
#define {upper_prefix}_GEN_H

#include <stddef.h>
#include <stdint.h>

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
        method_pool_size = 512 * 1024  # TODO: expose
        temp_pool_size = 128 * 1024  # TODO: expose
        out = """
// executorch_wrapper.cc

#include "executorch_wrapper.h"
#include "${prefix}_pte.h"

#include <executorch/runtime/runtime.h>
#include <executorch/runtime/core/program.h>
#include <executorch/runtime/core/method.h>
#include <executorch/runtime/core/tensor.h>
#include <executorch/runtime/core/evalue.h>

extern "C" {
extern const uint8_t model_pte[];
extern const size_t model_pte_len;
}

using namespace executorch::runtime;

namespace {

Program program;
Result<Method> method_result(nullptr);
Method* method = nullptr;

constexpr size_t METHOD_POOL_SIZE = ${method_pool_size};
constexpr size_t TEMP_POOL_SIZE   = ${temp_pool_size};

uint8_t method_pool[METHOD_POOL_SIZE];
uint8_t temp_pool[TEMP_POOL_SIZE];

MemoryAllocator method_allocator(METHOD_POOL_SIZE, method_pool);
MemoryAllocator temp_allocator(TEMP_POOL_SIZE, temp_pool);

HierarchicalAllocator planned_memory;
MemoryManager memory_manager(
    &method_allocator,
    &planned_memory,
    &temp_allocator
);

EValue* inputs;
EValue* outputs;

size_t num_inputs;
size_t num_outputs;

}

int model_init()
{
    runtime_init();

    BufferDataLoader loader(model_pte, model_pte_len);

    auto program_result = Program::load(&loader);
    if (!program_result.ok())
        return 1;

    program = std::move(program_result.get());

    auto method_name = program.get_method_name(0);
    if (!method_name.ok())
        return 1;

    method_result = program.load_method(
        *method_name,
        &memory_manager
    );

    if (!method_result.ok())
        return 1;

    method = &method_result.get();

    num_inputs = method->inputs_size();
    num_outputs = method->outputs_size();

    inputs = method_allocator.allocateList<EValue>(num_inputs);
    outputs = method_allocator.allocateList<EValue>(num_outputs);

    method->get_inputs(inputs, num_inputs);
    method->get_outputs(outputs, num_outputs);

    return 0;
}

void* model_input_ptr(int index)
{
    return inputs[index].toTensor().mutable_data_ptr<void>();
}

size_t model_input_size(int index)
{
    return inputs[index].toTensor().nbytes();
}

size_t model_inputs()
{
    return num_inputs;
}

void* model_output_ptr(int index)
{
    return outputs[index].toTensor().mutable_data_ptr<void>();
}

size_t model_output_size(int index)
{
    return outputs[index].toTensor().nbytes();
}

size_t model_outputs()
{
    return num_outputs;
}

int model_invoke()
{
    Error err = method->execute();

    if (err != Error::Ok)
        return 1;

    temp_allocator.reset(TEMP_POOL_SIZE, temp_pool);

    return 0;
}
"""
        out += fill(
            out,
            prefix=prefix,
            method_pool_size=method_pool_size,
            temp_pool_size=temp_pool_size,
        )
        return out
    wrapper = generate_wrapper()
    header = generate_header()
    return wrapper, header


class ExecutorchBackend(Backend):
    registry = {}

    # name = None
    name = "executorch"

    DEFAULTS = {
        "print_outputs": False,
    }

    # REQUIRED = {"executorch.src_dir", "executorch.build_dir"}
    REQUIRED = {"executorch.src_dir"}

    def __init__(self, output_format=None, hal_backend=None, hal_inline=False, features=None, config=None):
        super().__init__(framework="executorch", features=features, config=config)
        self.identifier = "model"

        self.model = None  # Actual filename!
        self.model_info = None
        self.input_shapes = None
        self.model_format = None
        # self.supported_formats = get_supported_formats_executorch()
        # self.supported_formats = [ModelFormats.TFLITE, ModelFormats.MLIR]
        self.supported_formats = [ModelFormats.PTE]
        # TODO: support PKL,...

        self.artifacts = []

    @property
    def executorch_src_dir(self):
        return Path(self.config["executorch.src_dir"])

    @property
    def pte_to_header_exe(self):
        return self.executorch_src_dir / "examples" / "riscv" / "executor_runner" / "pte_to_header.py"

    # @property
    # def executorch_build_dir(self):
    #     return self.config["executorch.build_dir"]

    @property
    def print_outputs(self):
        value = self.config["print_outputs"]
        return str2bool(value)

    def prepare_environment(self):
        env = os.environ.copy()
        pythonpath = env.get("PYTHONPATH", "")
        pythonpath = f"{self.executorch_src_dir}:{pythonpath}"
        # print("pythonpath", pythonpath)
        env["PYTHONPATH"] = pythonpath
        return env

    def generate_pte_header(self, pte_file: Path, out_file: Path, cwd=None):
        outdir = out_file.parent
        outfile = out_file.name
        args = [selt.pte_to_header, "--pte", pte_file, "--outdir", outdir, "--outfile", outfile]
        utils.execute(args, )
        env = self.prepare_environment()
        utils.python(*args, live=self.print_outputs, env=env, cwd=cwd)

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
            if self.model_format == "pte":
                pte_file = self.model
            else:
                pte_file = out_dir / f"{self.identifier}.pte"
                # TODO: generate
                raise NotImplementedError
                with open(pte_file, "rb") as f:
                    model_raw = f.read()
                artifacts.append(
                    Artifact(
                        pte_file.name,
                        raw=model_raw,
                        fmt=ArtifactFormat.BIN,
                    )
                )
            pte_header_file = out_dir / f"{self.identifier}_pte.h"
            self.generate_pte_header(pte_file, pte_header_file)
            artifacts.append(
                Artifact(
                    pte_header_file.name,
                    path=pte_header_file,
                    fmt=ArtifactFormat.SOURCE,
                )
            )
            wrapper_content, header_content = generate_executorch_wrapper(self.model_info, self.identifier)
            artifacts.append(
                Artifact(
                    "executorch_wrapper.cc",
                    content=wrapper_content,
                    fmt=ArtifactFormat.SOURCE,
                )
            )
            artifacts.append(
                Artifact(
                    "executorch_wrapper.h",
                    content=wrapper_header_content,
                    fmt=ArtifactFormat.SOURCE,
                )
            )
            stdout_artifact = Artifact(
                "executorch_out.log", content=out, fmt=ArtifactFormat.TEXT
            )
            artifacts.append(stdout_artifact)
        print("artifacts", artifacts)
        return {"default": artifacts}, {"default": metrics}

    def get_platform_defs(self, platform):
        ret = super().get_platform_defs(platform)
        return ret
