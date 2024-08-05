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
"""TODO"""

import string
from datetime import datetime
from math import ceil, log2

# TODO: use this
# from tvm.relay.backend.utils import mangle_module_name


def calc_pages(workspace_size, page_size=2**10):
    # Determine the number of required pages
    assert workspace_size >= 0, "Workspace size can not be negative"
    crtPageSizeLog2 = log2(page_size)
    assert crtPageSizeLog2 == int(crtPageSizeLog2), "Page size has to be a power of two"
    crtNumPages = ceil(workspace_size / page_size)
    return int(crtNumPages), int(crtPageSizeLog2)


def generate_wrapper_header():
    out = """#ifndef TVM_WRAPPER_H
#define TVM_WRAPPER_H

#include <stddef.h>

int TVMWrap_Init();
void *TVMWrap_GetInputPtr(int index);
size_t TVMWrap_GetInputSize(int index);
size_t TVMWrap_GetNumInputs();
int TVMWrap_Run();
void *TVMWrap_GetOutputPtr(int index);
size_t TVMWrap_GetOutputSize(int index);
size_t TVMWrap_GetNumOutputs();

#endif  // TVM_WRAPPER_H
"""
    return out


def generate_header():
    time = datetime.now()
    header = f"""// This file is generated. Do not edit.
// Generated on: {time}
"""
    return header


def generate_common_includes():
    return """
#include <stdlib.h>
#include <stdarg.h>
#include <dlpack/dlpack.h>
#include "tvm/runtime/crt/error_codes.h"
#include "tvm/runtime/c_runtime_api.h"
#include "printing.h"
#include "exit.h"
"""


def generate_graph_includes():
    out = generate_common_includes()
    out += """
#include "tvm/runtime/crt/packed_func.h"
#include "tvm/runtime/crt/crt.h"
#include "tvm/runtime/crt/graph_executor.h"
#include "tvm/runtime/crt/page_allocator.h"

"""
    return out


def generate_aot_includes(allocator):
    out = generate_common_includes()
    if allocator:
        out += '#include "tvm/runtime/crt/stack_allocator.h"\n'
    return out


def fill(template, **kwargs):
    return string.Template(template).substitute(**kwargs)


def getSizes(tensors):
    out = "size_t sizes[] = { "
    for t in tensors:
        out += str(t.size) + ", "
    out += "};"
    return out


def write_tvmrt_wrapper(path, graph, params, model_info, workspace_size):
    with open(path, "w") as f:
        text = generate_tvmrt_wrapper(graph, params, model_info, workspace_size)
        f.write(text)


def generate_tvmrt_wrapper(graph, params, model_info, workspace_size, debug_arena=False):
    crtNumPages, crtPageSizeLog2 = calc_pages(workspace_size)

    def escapeJson(j):
        return j.replace('"', '\\"').replace("\n", "\\\n")

    def toCArray(bin):
        result = ""
        for c in bin:
            result += hex(c) + ", "
        return result

    def getMeta(tensors, withNames=False):
        out = ""
        if withNames:
            out = "const char *names[] = { "
            for t in tensors:
                out += '"' + t.name + '", '
            out += "};\n    "

        out += "DLDataType dtypes[] = {"
        for t in tensors:
            if t.dtype == "float32":
                out += "{kDLFloat, 32, 1}"
            elif t.dtype == "uint8":
                out += "{kDLUInt, 8, 1}"
            elif t.dtype == "int8":
                out += "{kDLInt, 8, 1}"
            elif t.dtype == "uint64":
                out += "{kDLUInt, 64, 1}"
            elif t.dtype == "int64":
                out += "{kDLInt, 64, 1}"
            else:
                raise RuntimeError(f"Invalid type: {t.dtype}")
            out += ", "
        out += "};\n    "

        for i, t in enumerate(tensors):
            out += "int64_t shape_" + str(i) + "[] = { "
            for s in t.shape:
                out += str(s) + ", "
            out += "};\n    "
        out += "int64_t *shapes[] = { "
        for i, t in enumerate(tensors):
            out += "shape_" + str(i) + ", "
        out += "};\n"

        out += "    size_t ndims[] = { "
        for i, t in enumerate(tensors):
            out += str(len(t.shape)) + ", "
        out += "};\n    "

        for i, t in enumerate(tensors):
            out += "static uint8_t data_" + str(i) + "[" + str(t.size) + "];\n"
        out += "    uint8_t *data[] = { "
        for i, t in enumerate(tensors):
            out += "data_" + str(i) + ", "
        out += "};"

        return out

    out = ""
    out += generate_header()
    out += generate_graph_includes()
    out += 'const char * const g_graph = "' + escapeJson(graph) + '";\n'
    out += "const char g_params[] = { " + toCArray(params) + "\n};\n"
    out += "const uint64_t g_params_size = " + str(len(params)) + ";\n"

    mainCode = """


#define CRT_MEMORY_NUM_PAGES ${numPages}
#define CRT_MEMORY_PAGE_SIZE_LOG2 ${pageSizeLog2}
"""

    if debug_arena:  # This will enable the feature only if it is not overwritten by the user/compiler
        mainCode += """
#ifndef DEBUG_ARENA_USAGE
#define DEBUG_ARENA_USAGE 1
#endif
"""

    mainCode += """
#ifdef DEBUG_ARENA_USAGE
size_t max_arena_usage = 0;
#endif

static uint8_t g_crt_memory[CRT_MEMORY_NUM_PAGES * (1 << CRT_MEMORY_PAGE_SIZE_LOG2)];
static MemoryManagerInterface* g_memory_manager;

/*! \\brief macro to do C API call */
#define TVM_CCALL(func)                                                                   \\
    do {                                                                                  \\
        tvm_crt_error_t ret = (func);                                                     \\
        if (ret != kTvmErrorNoError) {                                                    \\
            TVMLogf("%s: %d: error: %s\\n", __FILE__, __LINE__, TVMGetLastError());       \\
            TVMPlatformAbort(ret);                                                        \\
        }                                                                                 \\
    } while (0)

TVMModuleHandle TVMArgs_AsModuleHandle(const TVMArgs* args, size_t index);

void __attribute__((noreturn)) TVMPlatformAbort(tvm_crt_error_t code)
{
    mlonmcu_exit(1);
}

void TVMLogf(const char* msg, ...)
{
    va_list args;
    va_start(args, msg);
    DBGPRINTF(msg, args);
    va_end(args);
}

tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr)
{
    tvm_crt_error_t ret = g_memory_manager->Allocate(g_memory_manager, num_bytes, dev, out_ptr);
#ifdef DEBUG_ARENA_USAGE
    // Use this to estimate the required number of pages
    // Run in DEBUG mode in insert value of MAX printed last into the following equation:
    // (This will round to the next power of 2 which might not be wanted!)
    // num_pages = 2**ceil(log2(MAX/page_size))
    size_t end = (size_t)(*out_ptr-(void*)g_crt_memory)+num_bytes;
    if (end > max_arena_usage)
    {
        max_arena_usage = end;
    }
#endif
    return ret;
}

tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev)
{
    return g_memory_manager->Free(g_memory_manager, ptr, dev);
}

tvm_crt_error_t TVMPlatformTimerStart()
{
    return kTvmErrorFunctionCallNotImplemented;
}

tvm_crt_error_t TVMPlatformTimerStop(double* elapsed_time_seconds)
{
  return kTvmErrorFunctionCallNotImplemented;
}

void *g_handle = NULL;

int TVMWrap_Init()
{
    int64_t device_type = kDLCPU;
    int64_t device_id = 0;

    TVMByteArray params;
    params.data = g_params;
    params.size = g_params_size;

    DLDevice dev;
    dev.device_type = (DLDeviceType)device_type;
    dev.device_id = device_id;

    // get pointers
    TVM_CCALL(PageMemoryManagerCreate(&g_memory_manager, g_crt_memory, sizeof(g_crt_memory),
                                      CRT_MEMORY_PAGE_SIZE_LOG2));
    TVM_CCALL(TVMInitializeRuntime());
    TVMPackedFunc pf;
    TVMArgs args = TVMArgs_Create(NULL, NULL, 0);
    TVM_CCALL(TVMPackedFunc_InitGlobalFunc(&pf, "runtime.SystemLib", &args));
    TVM_CCALL(TVMPackedFunc_Call(&pf));

    TVMModuleHandle mod_syslib = TVMArgs_AsModuleHandle(&pf.ret_value, 0);

    // run modules
    TVMGraphExecutor* graph_executor = NULL;
    TVM_CCALL(TVMGraphExecutor_Create(g_graph, mod_syslib, &dev, &graph_executor));
    TVM_CCALL(TVMGraphExecutor_LoadParams(graph_executor, params.data, params.size));

    //return graph_executor;
    g_handle = graph_executor;
    return 0;  // TODO
}

void *TVMWrap_GetInputPtr(int index)
{
    ${inMeta}

    DLTensor input;
    input.data = (void*)data[index];
    DLDevice device = {kDLCPU, 0};
    input.device = device;
    input.ndim = ndims[index];
    input.dtype = dtypes[index];
    input.shape = shapes[index];
    input.strides = NULL;
    input.byte_offset = 0;

    TVMGraphExecutor* graph_executor = (TVMGraphExecutor*)g_handle;
    TVMGraphExecutor_SetInput(graph_executor, names[index], &input);

    return data[index];
}

size_t TVMWrap_GetInputSize(int index)
{
    ${inSizes}

    return sizes[index];
}

size_t TVMWrap_GetNumInputs()
{
    return ${numInputs};
}

int TVMWrap_Run()
{
    TVMGraphExecutor* graph_executor = (TVMGraphExecutor*)g_handle;
    TVMGraphExecutor_Run(graph_executor);
#if DEBUG_ARENA_USAGE
    DBGPRINTF("\\nGraph executor arena max usage after model invocation: %lu bytes\\n", max_arena_usage);
#endif  // DEBUG_ARENA_USAGE
    return 0;  // TODO
}

void *TVMWrap_GetOutputPtr(int index)
{
    ${outMeta}

    DLTensor output;
    output.data = (void*)data[index];
    DLDevice device = {kDLCPU, 0};
    output.device = device;
    output.ndim = ndims[index];
    output.dtype = dtypes[index];
    output.shape = shapes[index];
    output.strides = NULL;
    output.byte_offset = 0;

    TVMGraphExecutor* graph_executor = (TVMGraphExecutor*)g_handle;
    TVMGraphExecutor_GetOutput(graph_executor, index, &output);

    return data[index];
}

size_t TVMWrap_GetOutputSize(int index)
{
    ${outSizes}

    return sizes[index];
}

size_t TVMWrap_GetNumOutputs()
{
    return ${numOutputs};
}
"""
    out += fill(
        mainCode,
        inMeta=getMeta(model_info.in_tensors, True),
        outMeta=getMeta(model_info.out_tensors),
        inSizes=getSizes(model_info.in_tensors),
        outSizes=getSizes(model_info.out_tensors),
        numInputs=len(model_info.in_tensors),
        numOutputs=len(model_info.out_tensors),
        numPages=crtNumPages,
        pageSizeLog2=crtPageSizeLog2,
    )
    return out


def generate_tvmaot_wrapper(model_info, workspace_size, mod_name, api="c", debug_arena=False):
    modPrefix = f"tvmgen_{mod_name}"

    def writeTensors(in_tensors, out_tensors, modPrefix, api):
        if api == "c":
            retStr = """
// Define data for input and output tensors
"""

            def writeTensorsHelper(tensors, prefix, out=False):
                lenTensors = len(tensors)
                direction = "out" if out else "in"
                ret = ""
                names = [f"{direction}put{i}_data" for i in range(lenTensors)]
                for i, t in enumerate(tensors):
                    ret += "char " + names[i] + "[" + str(t.size) + "];\n"
                ret += f"void* {direction}puts[] = {{" + ", ".join(names) + "};\n"
                ret += f"struct {prefix}_{direction}puts {prefix}_{direction}puts = {{" + "\n"
                for i, t in enumerate(tensors):
                    tensor_name = t.name.replace(":", "_").replace("/", "_").replace(".", "_").replace(";", "_")
                    ret += f"    .{tensor_name} = {names[i]}," + "\n"
                ret += "};\n"
                return ret

            retStr += writeTensorsHelper(in_tensors, modPrefix, False)
            retStr += writeTensorsHelper(out_tensors, modPrefix, True)
            return retStr
        elif api == "packed":
            retStr = """
// Define data for input and output tensors
"""

            def writeTensorsHelper(tensors, prefix, out=False):
                lenTensors = len(tensors)
                direction = "out" if out else "in"
                ret = ""
                names = [f"{direction}put{i}_data" for i in range(lenTensors)]
                for i, t in enumerate(tensors):
                    ret += "char " + names[i] + "[" + str(t.size) + "];\n"
                ret += f"void* {direction}puts[] = {{" + ", ".join(names) + "};\n"
                return ret

            retStr += writeTensorsHelper(in_tensors, modPrefix, False)
            retStr += writeTensorsHelper(out_tensors, modPrefix, True)
            return retStr
        else:
            raise RuntimeError("api has to be either 'c' or 'packed'")

    out = ""
    out += generate_header()
    includes = generate_aot_includes(workspace_size > 0)

    if api == "c":
        includes += '#include "${modPrefix}.h"\n'

    out += fill(includes, modPrefix=modPrefix)

    out += "\n"

    out += writeTensors(model_info.in_tensors, model_info.out_tensors, modPrefix, api)

    logging_code = """
void TVMLogf(const char* msg, ...)
{
    va_list args;
    va_start(args, msg);
    DBGPRINTF(msg, args);
    va_end(args);
}
"""

    out += logging_code

    if workspace_size > 0:
        workspace_code = """
#define WORKSPACE_SIZE (${workspaceBytes})
static uint8_t g_aot_memory[WORKSPACE_SIZE];
tvm_workspace_t app_workspace;
"""

        if debug_arena:  # This will enable the feature only if it is not overwritten by the user/compiler
            workspace_code += """
#ifndef DEBUG_ARENA_USAGE
#define DEBUG_ARENA_USAGE 1
#endif

#ifndef TVMAOT_DEBUG_ALLOCATIONS
#define TVMAOT_DEBUG_ALLOCATIONS 1
#endif
"""

        workspace_code += """
#ifdef DEBUG_ARENA_USAGE
size_t max_arena_usage = 0;
#endif

tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr)
{
#ifdef TVMAOT_DEBUG_ALLOCATIONS
    if (num_bytes > (app_workspace.workspace + app_workspace.workspace_size - app_workspace.next_alloc))
    {
      TVMLogf("TVMPlatformMemoryAllocate(%lu): Allocation would overflow arena!\\n", num_bytes);
      return kTvmErrorPlatformNoMemory;
    }
#endif
    tvm_crt_error_t ret = StackMemoryManager_Allocate(&app_workspace, num_bytes, out_ptr);
#ifdef DEBUG_ARENA_USAGE
  // Use this to estimate the required number of bytes for the arena
  size_t end = app_workspace.next_alloc-app_workspace.workspace;
  if (end > max_arena_usage)
  {
    max_arena_usage = end;
  }
#endif
    return ret;
}
tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev)
{
#ifdef TVMAOT_DEBUG_ALLOCATIONS
    if ((uint8_t*)ptr < app_workspace.workspace || (uint8_t*)ptr >= app_workspace.next_alloc)
    {
      TVMLogf("TVMPlatformMemoryFree(%p): Invalid Memory region to be free'd!\\n", ptr);
      return kTvmErrorPlatformNoMemory;
    }
#endif
    return StackMemoryManager_Free(&app_workspace, ptr);
}
"""
        out += fill(workspace_code, workspaceBytes=int(workspace_size))
    else:
        workspace_code = """
tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr)
{
    return kTvmErrorFunctionCallNotImplemented;
}
tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev)
{
    return kTvmErrorFunctionCallNotImplemented;
}
"""
        out += workspace_code

    mainCode = ""
    if api == "packed":
        mainCode += (
            "int32_t ${modPrefix}_run(void* args, void* type_code, int num_args,"
            + " void* out_value, void* out_type_code, void* resource_handle);\n"
        )

    mainCode += """
void __attribute__((noreturn)) TVMPlatformAbort(tvm_crt_error_t code)
{
    mlonmcu_exit(1);
}

TVM_DLL int TVMFuncRegisterGlobal(const char* name, TVMFunctionHandle f, int override)
{
    return 0;
}

int TVMWrap_Init()
{
"""
    if workspace_size > 0:
        mainCode += "    StackMemoryManager_Init(&app_workspace, g_aot_memory, WORKSPACE_SIZE);"
    mainCode += """
    return 0;  // TODO
}

void *TVMWrap_GetInputPtr(int index)
{
    return inputs[index];
}

size_t TVMWrap_GetInputSize(int index)
{
    ${inSizes}

    return sizes[index];
}

size_t TVMWrap_GetNumInputs()
{
    return ${numInputs};
}

int TVMWrap_Run()
{"""
    if api == "c":
        mainCode += """
    int ret_val = ${modPrefix}_run(&${modPrefix}_inputs, &${modPrefix}_outputs);
    if (ret_val)
    {
        TVMPlatformAbort(kTvmErrorPlatformCheckFailure);
    }
"""
    elif api == "packed":
        mainCode += """
    static DLDevice fake_device = {kDLCPU, 0};
    static int64_t fake_dims = 0;
    static int64_t fake_shape = {0};

    DLTensor tensors[${numInputs} + ${numOutputs}];
    TVMValue values[${numInputs} + ${numOutputs}];
    int32_t typeids[${numInputs} + ${numOutputs}];

    for (size_t i = 0; i < ${numInputs}+${numOutputs}; i++)
    {
        tensors[i].device = fake_device;
        tensors[i].data = (i < ${numInputs}) ? inputs[i] : outputs[i - ${numInputs}];
        tensors[i].shape = &fake_shape;
        tensors[i].ndim = fake_dims;
        tensors[i].byte_offset = 0;
        tensors[i].strides = NULL;
        values[i].v_handle = &tensors[i];
    }

    int ret_val = ${modPrefix}_run(values, typeids, 0, NULL, 0, NULL);
    if (ret_val)
    {
        TVMPlatformAbort(kTvmErrorPlatformCheckFailure);
    }
    return 0;

"""
    else:
        raise RuntimeError("api can only be 'c' or 'packed'")

    if workspace_size > 0:
        mainCode += """
#if DEBUG_ARENA_USAGE
    DBGPRINTF("\\nAoT executor arena max usage after model invocation: %lu bytes\\n", max_arena_usage);
#endif  // DEBUG_ARENA_USAGE
"""
    mainCode += """
    return 0;  // TODO
}

void *TVMWrap_GetOutputPtr(int index)
{
    return outputs[index];
}

size_t TVMWrap_GetOutputSize(int index)
{
    ${outSizes}

    return sizes[index];
}

size_t TVMWrap_GetNumOutputs()
{
    return ${numOutputs};
}
"""
    out += fill(
        mainCode,
        inSizes=getSizes(model_info.in_tensors),
        outSizes=getSizes(model_info.out_tensors),
        numInputs=len(model_info.in_tensors),
        numOutputs=len(model_info.out_tensors),
        modPrefix=modPrefix,
    )
    return out


def write_tvmaot_wrapper(path, model_info, workspace_size, mod_name, api="c"):
    with open(path, "w") as f:
        text = generate_tvmaot_wrapper(model_info, workspace_size, mod_name, api=api)
        f.write(text)
