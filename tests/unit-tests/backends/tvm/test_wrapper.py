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
"""Unit tests for the tvm wrapper generator."""
import pytest

from mlonmcu.flow.tvm.backend import wrapper
from mlonmcu.flow.tvm.backend.model_info import TensorInfo, ModelInfo

DUMMY_GRAPH_JSON = "{}"
DUMMY_PARAMS = bytes()

MODEL_INFO_1 = ModelInfo(
    [
        TensorInfo("input", (1024,), "int8"),
    ],
    [
        TensorInfo("output", (8,), "int8"),
    ],
)
MODEL_INFO_2 = ModelInfo(
    [
        TensorInfo("input", (10, 10), "float32"),
        TensorInfo("input2", (1, 16), "float32"),
    ],
    [
        TensorInfo("output", (8,), "float32"),
        TensorInfo("output2", (2, 4), "float32"),
    ],
)


def _check(out, expected_lines):
    out_lines = out.split("\n")[2:]
    for expected_line in expected_lines:
        assert expected_line in out_lines


def test_calc_pages():
    # valid
    assert wrapper.calc_pages(2**15, page_size=2**10) == (32, 10)
    assert wrapper.calc_pages((2**15) - 1, page_size=2**10) == (32, 10)
    assert wrapper.calc_pages(2**15, page_size=2**11) == (16, 11)
    # invalid
    with pytest.raises(AssertionError):
        wrapper.calc_pages(-1)
    with pytest.raises(AssertionError):
        wrapper.calc_pages(2**15, page_size=123)


@pytest.mark.parametrize("model_info", [MODEL_INFO_1, MODEL_INFO_2])
@pytest.mark.parametrize("workspace_size", [2**15])
@pytest.mark.parametrize("debug_arena", [False, True])
def test_wrapper_graph(model_info, workspace_size, debug_arena):
    out = wrapper.generate_tvmrt_wrapper(DUMMY_GRAPH_JSON, DUMMY_PARAMS, model_info, workspace_size, debug_arena)
    expected_lines = [
        "#include <stdlib.h>",
        "#include <stdarg.h>",
        "#include <dlpack/dlpack.h>",
        '#include "tvm/runtime/c_runtime_api.h"',
        '#include "tvm/runtime/crt/error_codes.h"',
        '#include "printing.h"',
    ]
    expected_lines.extend(
        [
            '#include "tvm/runtime/crt/packed_func.h"',
            '#include "tvm/runtime/crt/crt.h"',
            '#include "tvm/runtime/crt/graph_executor.h"',
            '#include "tvm/runtime/crt/page_allocator.h"',
        ]
    )
    # expected_lines.extend(
    #     [
    #         "char input0_data[400];",
    #         "char input1_data[64];",
    #         "char output0_data[32];",
    #         "char output1_data[32];",
    #         "void* inputs[] = {input0_data, input1_data};",
    #         "void* outputs[] = {output0_data, output1_data};",
    #     ]
    # )
    expected_lines.extend(
        [
            "tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr)",
            "    tvm_crt_error_t ret = g_memory_manager->Allocate(g_memory_manager, num_bytes, dev, out_ptr);",
            "tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev)",
            "    return g_memory_manager->Free(g_memory_manager, ptr, dev);",
            "    TVM_CCALL(PageMemoryManagerCreate(&g_memory_manager, g_crt_memory, sizeof(g_crt_memory),",
            "                                      CRT_MEMORY_PAGE_SIZE_LOG2));",
            "    TVM_CCALL(TVMInitializeRuntime());",
            '    TVM_CCALL(TVMPackedFunc_InitGlobalFunc(&pf, "runtime.SystemLib", &args));',
            "    TVM_CCALL(TVMPackedFunc_Call(&pf));",
            "    TVM_CCALL(TVMGraphExecutor_Create(g_graph, mod_syslib, &dev, &graph_executor));",
            "    TVM_CCALL(TVMGraphExecutor_LoadParams(graph_executor, params.data, params.size));",
            "#define CRT_MEMORY_NUM_PAGES 32",
            "#define CRT_MEMORY_PAGE_SIZE_LOG2 10",
            "static uint8_t g_crt_memory[CRT_MEMORY_NUM_PAGES * (1 << CRT_MEMORY_PAGE_SIZE_LOG2)];",
            "static MemoryManagerInterface* g_memory_manager;",
        ]
    )
    expected_lines.extend(
        [
            "void __attribute__((noreturn)) TVMPlatformAbort(tvm_crt_error_t code)",
            "int TVMWrap_Init()",
        ]
    )
    expected_lines.extend(
        [
            "void *TVMWrap_GetInputPtr(int index)",
            # "    return inputs[index];",
            "size_t TVMWrap_GetInputSize(int index)",
            # "    size_t sizes[] = { 400, 64, };",
            # "    return sizes[index];",
            "size_t TVMWrap_GetNumInputs()",
            # "    return 2;",
        ]
    )
    expected_lines.extend(
        [
            "int TVMWrap_Run()",
        ]
    )
    expected_lines.extend(
        [
            # "    if (ret_val)",
            # "        TVMPlatformAbort(kTvmErrorPlatformCheckFailure);",
        ]
    )
    if debug_arena and workspace_size > 0:
        expected_lines.extend(
            [
                "#if DEBUG_ARENA_USAGE",
                (
                    '    DBGPRINTF("\\nGraph executor arena max usage after model invocation: %lu bytes\\n",'
                    " max_arena_usage);"
                ),
                "#endif  // DEBUG_ARENA_USAGE",
            ]
        )
    expected_lines.extend(
        [
            "void *TVMWrap_GetOutputPtr(int index)",
            # "    return outputs[index];",
            "size_t TVMWrap_GetOutputSize(int index)",
            # "    size_t sizes[] = { 32, 32, };",
            # "    return sizes[index];",
            "size_t TVMWrap_GetNumOutputs()",
            # "    return 2;",
        ]
    )
    _check(out, expected_lines)


@pytest.mark.parametrize("model_info", [MODEL_INFO_1, MODEL_INFO_2])
@pytest.mark.parametrize("workspace_size", [0, 2**15])
@pytest.mark.parametrize("mod_name", ["default", "my_module"])
@pytest.mark.parametrize("api", ["c", "packed"])
@pytest.mark.parametrize("debug_arena", [False, True])
def test_wrapper_aot(model_info, workspace_size, mod_name, api, debug_arena):
    out = wrapper.generate_tvmaot_wrapper(model_info, workspace_size, mod_name, api, debug_arena)
    expected_lines = [
        "#include <stdlib.h>",
        "#include <stdarg.h>",
        "#include <dlpack/dlpack.h>",
        '#include "tvm/runtime/c_runtime_api.h"',
        '#include "tvm/runtime/crt/error_codes.h"',
        '#include "printing.h"',
    ]
    if workspace_size != 0:
        expected_lines.append('#include "tvm/runtime/crt/stack_allocator.h"')
    # expected_lines.extend(
    #     [
    #         "char input0_data[400];",
    #         "char input1_data[64];",
    #         "char output0_data[32];",
    #         "char output1_data[32];",
    #         "void* inputs[] = {input0_data, input1_data};",
    #         "void* outputs[] = {output0_data, output1_data};",
    #     ]
    # )
    if api == "packed":
        pass
        # expected_lines.extend(
        #     [
        #         "void* inputs[] = {input0_data, input1_data};",
        #         "void* outputs[] = {output0_data, output1_data};",
        #     ]
        # )
    elif api == "packed":
        expected_lines.extend(
            [
                f'#include "tvmgen_{mod_name}.h"',
                f"struct tvmgen_{mod_name}_inputs tvmgen_{mod_name}_inputs = {{",
                f"struct tvmgen_{mod_name}_outputs tvmgen_{mod_name}_outputs = {{",
            ]
        )
    if workspace_size > 0:
        expected_lines.extend(
            [
                f"#define WORKSPACE_SIZE ({int(workspace_size)})",
                "static uint8_t g_aot_memory[WORKSPACE_SIZE];",
                "tvm_workspace_t app_workspace;",
                "tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr)",
                "    tvm_crt_error_t ret = StackMemoryManager_Allocate(&app_workspace, num_bytes, out_ptr);",
                "tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev)",
                "    return StackMemoryManager_Free(&app_workspace, ptr);",
                "    StackMemoryManager_Init(&app_workspace, g_aot_memory, WORKSPACE_SIZE);",
            ]
        )
    expected_lines.extend(
        [
            "void __attribute__((noreturn)) TVMPlatformAbort(tvm_crt_error_t code)",
            "TVM_DLL int TVMFuncRegisterGlobal(const char* name, TVMFunctionHandle f, int override)",
            "int TVMWrap_Init()",
        ]
    )
    expected_lines.extend(
        [
            "void *TVMWrap_GetInputPtr(int index)",
            "    return inputs[index];",
            "size_t TVMWrap_GetInputSize(int index)",
            # "    size_t sizes[] = { 400, 64, };",
            "    return sizes[index];",
            "size_t TVMWrap_GetNumInputs()",
            # "    return 2;",
        ]
    )
    if api == "packed":
        expected_lines.extend(
            [
                (
                    f"int32_t tvmgen_{mod_name}_run(void* args, void* type_code, int num_args, void* out_value, void*"
                    " out_type_code, void* resource_handle);"
                ),
                "int TVMWrap_Run()",
                "    static DLDevice fake_device = {kDLCPU, 0};",
                "    static int64_t fake_dims = 0;",
                "    static int64_t fake_shape = {0};",
                # "    DLTensor tensors[2 + 2];",
                # "    TVMValue values[2 + 2];",
                # "    int32_t typeids[2 + 2];",
                # "    for (size_t i = 0; i < 2+2; i++) {",
                "        tensors[i].device = fake_device;",
                # "        tensors[i].data = (i < 2) ? inputs[i] : outputs[i - 2];",
                "        tensors[i].shape = &fake_shape;",
                "        tensors[i].ndim = fake_dims;",
                "        tensors[i].byte_offset = 0;",
                "        tensors[i].strides = NULL;",
                "        values[i].v_handle = &tensors[i];",
                f"    int ret_val = tvmgen_{mod_name}_run(values, typeids, 0, NULL, 0, NULL);",
            ]
        )
    elif api == "c":
        expected_lines.extend(
            [
                "int TVMWrap_Run()",
                f"    int ret_val = tvmgen_{mod_name}_run(&tvmgen_{mod_name}_inputs, &tvmgen_{mod_name}_outputs);",
            ]
        )
    expected_lines.extend(
        [
            "    if (ret_val)",
            "        TVMPlatformAbort(kTvmErrorPlatformCheckFailure);",
        ]
    )
    if debug_arena and workspace_size > 0:
        expected_lines.extend(
            [
                "#if DEBUG_ARENA_USAGE",
                (
                    '    DBGPRINTF("\\nAoT executor arena max usage after model invocation: %lu bytes\\n",'
                    " max_arena_usage);"
                ),
                "#endif  // DEBUG_ARENA_USAGE",
            ]
        )
    expected_lines.extend(
        [
            "void *TVMWrap_GetOutputPtr(int index)",
            "    return outputs[index];",
            "size_t TVMWrap_GetOutputSize(int index)",
            # "    size_t sizes[] = { 32, 32, };",
            "    return sizes[index];",
            "size_t TVMWrap_GetNumOutputs()",
            # "    return 2;",
        ]
    )
    _check(out, expected_lines)
