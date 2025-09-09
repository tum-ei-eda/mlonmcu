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
    get_supported_formats_iree,
    # get_model_format,
)
from mlonmcu.flow.tvm.backend.wrapper import getSizes
from mlonmcu.target.metrics import Metrics
from mlonmcu.artifact import Artifact, ArtifactFormat

# from mlonmcu.models.model import ModelFormats

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


def get_iree_compile_hal_backend_target_args(hal_backend, target_details):
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
        f"--iree-llvmcpu-enable-scalable-vectorization={int(target_scalable_vector)}",
        f"--iree-llvmcpu-fail-on-large-vector={1-int(supports_vectorization)}",
        *([f"--iree-llvmcpu-loop-unrolling={int(loop_unroll)}"] if loop_unroll is not None else []),
    ]
    return ret


def generate_iree_wrapper(
    model_info, identifier: str, use_emitc: bool = False, vmvx: bool = False, translated: bool = False
):
    main_func_name = model_info.main_func_name
    print("main_func_name", main_func_name)
    assert main_func_name is not None
    # identifier2 = "module_linked" if translated else f"{main_func_name}_dispatch_0"
    identifier2 = "model_linked" if translated else f"{main_func_name}_dispatch_0"
    print("identifier2", identifier2)
    inSizes = getSizes(model_info.in_tensors)
    outSizes = getSizes(model_info.out_tensors)
    numInputs = len(model_info.in_tensors)
    numOutputs = len(model_info.out_tensors)

    def writeTensors(in_tensors, out_tensors):
        retStr = """
// Define data for input and output tensors
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

    tensorBufs = writeTensors(model_info.in_tensors, model_info.out_tensors)

    def getIOSetupCode(in_tensors, out_tensors, use_emitc: bool = False):
        lookup_hal_element_type = {
            "int8": "IREE_HAL_ELEMENT_TYPE_SINT_8",
            "int32": "IREE_HAL_ELEMENT_TYPE_SINT_32",
            "float32": "IREE_HAL_ELEMENT_TYPE_FLOAT_32",
        }
        ret = """
// TODO: setup IO
"""
        num_inputs = len(in_tensors)
        num_outputs = len(out_tensors)
        for i, in_tensor in enumerate(in_tensors):
            shape = in_tensor.shape
            shape = [x if x is not None else 1 for x in shape]
            shape_str = "{" + ", ".join(map(str, shape)) + "}"
            dtype = in_tensor.dtype
            hal_element_type = lookup_hal_element_type[dtype]
            ret += f"""
    iree_hal_dim_t shape{i}[{len(shape)}] = {shape_str};  // TODO: fixed size?
    iree_hal_buffer_view_t *arg{i}_buffer_view = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_view_allocate_buffer_copy(
        device, iree_hal_device_allocator(device), IREE_ARRAYSIZE(shape{i}), shape{i},
        {hal_element_type}, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        (iree_hal_buffer_params_t){{
            .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
            .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
        }},
        iree_make_const_byte_span(inputs[{i}], sizeof(*inputs[{i}])), &arg{i}_buffer_view));
"""
        if not use_emitc:
            ret += f"""
  // Setup call inputs with our buffers.
  IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                           /*capacity=*/{num_inputs},
                                           iree_allocator_system(), &inputs_),
                       "can't allocate input vm list");
"""
        for i, in_tensor in enumerate(in_tensors):
            if not use_emitc:
                ret += f"""
  iree_vm_ref_t arg{i}_buffer_view_ref = iree_hal_buffer_view_move_ref(arg{i}_buffer_view);
  IREE_RETURN_IF_ERROR(iree_vm_list_push_ref_move(inputs_, &arg{i}_buffer_view_ref));
"""
            else:
                ret += f"""
  IREE_RETURN_IF_ERROR(iree_runtime_call_inputs_push_back_buffer_view(&call, arg{i}_buffer_view));
  iree_hal_buffer_view_release(arg{i}_buffer_view);
"""
        if not use_emitc:
            ret += f"""
  // Prepare outputs list to accept the results from the invocation.
  // The output vm list is allocated statically.
  IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                           /*capacity=*/{num_outputs},
                                           iree_allocator_system(), &outputs_),
                       "can't allocate output vm list");
"""
        return ret

    # setupInputsOutputs = getIOSetupCode(model_info.in_tensors, model_info.out_tensors, use_emitc=use_emitc)
    setupInputsOutputs = getIOSetupCode(model_info.in_tensors, model_info.out_tensors, use_emitc=False)

    def getCopyOutputsCode(out_tensors):
        assert len(out_tensors) == 1
        ret = """
  iree_hal_buffer_view_t *ret_buffer_view =
      iree_vm_list_get_buffer_view_assign(outputs_, 0);
  if (ret_buffer_view == NULL) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "can't find return buffer view");
  }
  // printf("R\\n");

  // Read back the results and ensure we got the right values.
  IREE_RETURN_IF_ERROR(iree_hal_device_transfer_d2h(
      device, iree_hal_buffer_view_buffer(ret_buffer_view), 0, outputs[0],
      sizeof(*outputs[0]), IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout()));
"""
        return ret

    copyOutputs = getCopyOutputsCode(model_info.out_tensors)

    wrapper_main = (
        """
#include <stddef.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/module.h"
#include "iree/modules/hal/inline/module.h"
#include "iree/modules/hal/loader/module.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"

// Initial buffer contents for 4 * 2 = 8.
// const int32_t kInt4[] = {4, 4, 4, 4};
// const int32_t kInt2[] = {2, 2, 2, 2};
// int32_t results[] = {0, 0, 0, 0};
"""
        + tensorBufs
        + """


// A function to create the HAL device from the different backend targets.
// The HAL device is returned based on the implementation, and it must be
// released by the caller.
extern iree_status_t create_sample_device(
    iree_allocator_t host_allocator, iree_hal_device_t** out_device,
    iree_hal_executable_loader_t** loader);

// A function to create the bytecode or C module.
extern iree_status_t create_module(iree_vm_instance_t* instance,
                                   iree_vm_module_t** out_module);

// static globals
static iree_vm_instance_t *instance = NULL;
static iree_hal_device_t *device = NULL;
static iree_vm_list_t *outputs_ = NULL;
static iree_vm_list_t *inputs_ = NULL;
static iree_vm_function_t main_function;
static iree_vm_context_t *context = NULL;

iree_status_t Prepare(void) {
  printf("A\\n");
  IREE_RETURN_IF_ERROR(iree_vm_instance_create(
      IREE_VM_TYPE_CAPACITY_DEFAULT, iree_allocator_system(), &instance));
  // IREE_RETURN_IF_ERROR(iree_hal_module_register_all_types(instance));
#if defined(BUILD_INLINE_HAL)
  IREE_RETURN_IF_ERROR(iree_hal_module_register_inline_types(instance));
#elif defined(BUILD_LOADER_HAL)
  IREE_RETURN_IF_ERROR(iree_hal_module_register_loader_types(instance));
#else
  IREE_RETURN_IF_ERROR(iree_hal_module_register_all_types(instance));
#endif
  printf("B\\n");

  iree_hal_executable_loader_t* loader = NULL;
  IREE_RETURN_IF_ERROR(
      create_sample_device(iree_allocator_system(), &device, &loader),
      "create device");
  printf("C\\n");

#if defined(BUILD_INLINE_HAL) || defined(BUILD_LOADER_HAL)
  // Create hal_inline_module
  iree_vm_module_t* hal_inline_module = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_inline_module_create(
      instance, IREE_HAL_INLINE_MODULE_FLAG_NONE,
      iree_hal_module_debug_sink_stdio(stderr),
      iree_hal_device_allocator(device), iree_allocator_system(),
      &hal_inline_module));
#endif
  printf("D\\n");


  iree_vm_module_t *module = NULL;
  IREE_RETURN_IF_ERROR(create_module(instance, &module));
  printf("E\\n");

  // iree_vm_module_t *hal_module = NULL;
  // IREE_RETURN_IF_ERROR(iree_hal_module_create(
  //     instance, /*device_count=*/1, &device, IREE_HAL_MODULE_FLAG_SYNCHRONOUS,
  //     iree_hal_module_debug_sink_stdio(stderr), iree_allocator_system(),
  //     &hal_module));
#if defined(BUILD_INLINE_HAL)
  iree_vm_module_t* modules[] = {hal_inline_module, module};
#elif defined(BUILD_LOADER_HAL)
  // Create hal_loader_module
  iree_vm_module_t* hal_loader_module = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_loader_module_create(
      instance, IREE_HAL_MODULE_FLAG_NONE,
      /*loader_count=*/1, &loader, iree_allocator_system(),
      &hal_loader_module));
  iree_vm_module_t* modules[] = {hal_inline_module, hal_loader_module, module};
#else
  // Create hal_module
  iree_vm_module_t* hal_module = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_module_create(
      instance, /*device_count=*/1, &device, IREE_HAL_MODULE_FLAG_SYNCHRONOUS,
      iree_hal_module_debug_sink_stdio(stderr), iree_allocator_system(),
      &hal_module));

  iree_vm_module_t* modules[] = {hal_module, module};
#endif
  iree_hal_executable_loader_release(loader);
  printf("F\\n");

  // Allocate a context that will hold the module state across invocations.
  // iree_vm_module_t *modules[] = {hal_module, module};
  IREE_RETURN_IF_ERROR(iree_vm_context_create_with_modules(
      instance, IREE_VM_CONTEXT_FLAG_NONE, IREE_ARRAYSIZE(modules), &modules[0],
      iree_allocator_system(), &context));
  printf("G\\n");
  // iree_vm_module_release(hal_module);
#if defined(BUILD_INLINE_HAL) || defined(BUILD_LOADER_HAL)
  iree_vm_module_release(hal_inline_module);
#else
  iree_vm_module_release(hal_module);
#endif
  printf("H\\n");

#if defined(BUILD_LOADER_HAL)
  iree_vm_module_release(hal_loader_module);
#endif
  iree_vm_module_release(module);
  printf("I\\n");

  // Lookup the entry point function.
  // Note that we use the synchronous variant which operates on pure type/shape
  // erased buffers.
  const char kMainFunctionName[] = \"module."""
        + main_func_name
        + """\";
  IREE_RETURN_IF_ERROR(iree_vm_context_resolve_function(
      context, iree_make_cstring_view(kMainFunctionName), &main_function));
  printf("J\\n");

  // Allocate buffers in device-local memory so that if the device has an
  // independent address space they live on the fast side of the fence.
  """
        + setupInputsOutputs
        + """
  printf("KLMNOP\\n");

  return iree_ok_status();
}

iree_status_t Run(void) {
  // Synchronously invoke the function.
  // IREE_RETURN_IF_ERROR(iree_vm_invoke(
  iree_status_t status = iree_vm_invoke(
      context, main_function, IREE_VM_INVOCATION_FLAG_NONE,
      /*policy=*/NULL, inputs_, outputs_, iree_allocator_system());
      ///*policy=*/NULL, inputs_, outputs_, iree_allocator_system()));
  iree_status_fprint(stdout, status);
  // return iree_ok_status();
  return status;
}

iree_status_t Cleanup(void) {
  printf("Q\\n");

  // Get the result buffers from the invocation.
  """
        + copyOutputs
        + """
  printf("S\\n");
  // for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(results); ++i) {
  //   if (results[i] != 8) {
  //     return iree_make_status(IREE_STATUS_UNKNOWN, "result mismatches");
  //   }
  // }

  // Print statistics (no-op if statistics are not enabled).
  iree_hal_allocator_statistics_fprint(stdout,
                                       iree_hal_device_allocator(device));
  printf("T\\n");

  iree_vm_list_release(inputs_);
  iree_vm_list_release(outputs_);
  iree_hal_device_release(device);
  iree_vm_context_release(context);
  iree_vm_instance_release(instance);
  printf("U\\n");
  return iree_ok_status();
}
"""
    )

    epilog = (
        """
int IREE_Init()
{
    iree_status_t status = Prepare();
    return !iree_status_is_ok(status);
}

int IREE_Deinit()
{
    iree_status_t status = Cleanup();
    return !iree_status_is_ok(status);
}

void *IREE_GetInputPtr(int index)
{
    // void *data[] = {&kInt2[0], &kInt4[0]};
    // return data[index];
    return inputs[index];
}

size_t IREE_GetInputSize(int index)
{
    // const size_t sizes[] = {16, 16};
    """
        + inSizes
        + """
    return sizes[index];
}

size_t IREE_GetNumInputs()
{
    return """
        + str(numInputs)
        + """;
}

int IREE_Run()
{
    iree_status_t status = Run();
    return !iree_status_is_ok(status);
}

void *IREE_GetOutputPtr(int index)
{
    // void *data[] = {&results[0]};
    // return data[index];
    return outputs[index];
}

size_t IREE_GetOutputSize(int index)
{
    // const size_t sizes[] = {16};
    """
        + outSizes
        + """
    return sizes[index];
}

size_t IREE_GetNumOutputs()
{
    return """
        + str(numOutputs)
        + """;
}
"""
    )
    header = """#ifndef IREE_WRAPPER_H
#define IREE_WRAPPER_H

#include <stddef.h>

int IREE_Init();
int IREE_Deinit();
void *IREE_GetInputPtr(int index);
size_t IREE_GetInputSize(int index);
size_t IREE_GetNumInputs();
int IREE_Run();
void *IREE_GetOutputPtr(int index);
size_t IREE_GetOutputSize(int index);
size_t IREE_GetNumOutputs();

#endif  // IREE_WRAPPER_H
"""
    sync_static = (
        """
#include <stddef.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/local_sync/sync_device.h"
#include "iree/hal/local/executable_loader.h"
#include "iree/hal/local/loaders/static_library_loader.h"

#include \""""
        + identifier
        + """_static_lib.h\"

iree_status_t create_sample_device(iree_allocator_t host_allocator,
                                   iree_hal_device_t **out_device,
                                   iree_hal_executable_loader_t** loader) {

  // Set parameters for the device created in the next step.
  iree_hal_sync_device_params_t params;
  iree_hal_sync_device_params_initialize(&params);

  const iree_hal_executable_library_query_fn_t libraries[] = {
      """
        + identifier2
        + """_library_query,
  };

  iree_status_t status = iree_hal_static_library_loader_create(
      IREE_ARRAYSIZE(libraries), libraries,
      iree_hal_executable_import_provider_null(), host_allocator, loader);

  // Use the default host allocator for buffer allocations.
  iree_string_view_t identifier = iree_make_cstring_view("local-sync");
  iree_hal_allocator_t* device_allocator = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_create_heap(identifier, host_allocator,
                                            host_allocator, &device_allocator);
  }

  if (iree_status_is_ok(status)) {
    // Create the synchronous device
    status = iree_hal_sync_device_create(
        identifier, &params, /*loader_count=*/1, loader, device_allocator,
        host_allocator, out_device);
  }

  iree_hal_allocator_release(device_allocator);
  return status;
}
"""
    )
    sync_vmvx = """
#include <stddef.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/local_sync/sync_device.h"
#include "iree/hal/local/executable_loader.h"
#include "iree/hal/local/loaders/vmvx_module_loader.h"

iree_status_t create_sample_device(iree_allocator_t host_allocator,
                                   iree_hal_device_t** out_device,
                                   iree_hal_executable_loader_t** loader) {
  // Set parameters for the device created in the next step.
  iree_hal_sync_device_params_t params;
  iree_hal_sync_device_params_initialize(&params);

  iree_vm_instance_t* instance = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                               host_allocator, &instance));

  iree_status_t status = iree_hal_vmvx_module_loader_create(
      instance, /*user_module_count=*/0, /*user_modules=*/NULL, host_allocator,
      loader);
  iree_vm_instance_release(instance);

  // Use the default host allocator for buffer allocations.
  iree_string_view_t identifier = iree_make_cstring_view("local-sync");
  iree_hal_allocator_t* device_allocator = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_create_heap(identifier, host_allocator,
                                            host_allocator, &device_allocator);
  }

  if (iree_status_is_ok(status)) {
    // Create the synchronous device.
    status = iree_hal_sync_device_create(
        identifier, &params, /*loader_count=*/1, loader, device_allocator,
        host_allocator, out_device);
  }

  iree_hal_allocator_release(device_allocator);
  return status;
}
"""
    utils_vmvx = (
        """
#include <stdio.h>

#include "iree/vm/bytecode/module.h"

#include \""""
        + identifier
        + """.h\"

// A function to create a VMVX bytecode module.
iree_status_t create_module(iree_vm_instance_t* instance,
                            iree_vm_module_t** out_module) {
  const struct iree_file_toc_t *module_file_toc = """
        + identifier
        + """_create();
  iree_const_byte_span_t module_data =
      iree_make_const_byte_span(module_file_toc->data, module_file_toc->size);
  return iree_vm_bytecode_module_create(instance, module_data,
                                        iree_allocator_null(),
                                        iree_allocator_system(), out_module);
}
"""
    )
    utils_bytecode = (
        """
#include <stdio.h>

#include "iree/vm/bytecode/module.h"

#include \""""
        + identifier
        + """.h\"

// A function to create the bytecode module.
iree_status_t create_module(iree_vm_instance_t* instance,
                            iree_vm_module_t** out_module) {
  const struct iree_file_toc_t *module_file_toc = """
        + identifier
        + """_create();
  iree_const_byte_span_t module_data =
      iree_make_const_byte_span(module_file_toc->data, module_file_toc->size);
  return iree_vm_bytecode_module_create(
      instance, module_data, iree_allocator_null(),
      iree_vm_instance_allocator(instance), out_module);
}
"""
    )
    utils_emitc = (
        """
#include <stdio.h>

#include \""""
        + identifier
        + """_emitc.h\"

// A function to create the C module.
iree_status_t create_module(iree_vm_instance_t* instance,
                            iree_vm_module_t** out_module) {
  return module_create(instance, iree_vm_instance_allocator(instance),
                       out_module);
}
"""
    )
    prolog = ""
    sync = sync_vmvx if vmvx else sync_static
    wrapper = prolog + wrapper_main + epilog
    utils_ = utils_vmvx if vmvx else (utils_emitc if use_emitc else utils_bytecode)
    return wrapper, header, sync, utils_


class IREEBackend(Backend):
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
        # "iree_version": None,  # TODO: auto?
        "target_vector_width": None,
        "target_scalable_vectorization": None,
        "loop_unroll": True,
    }

    OPTIONAL = {"iree.version"}

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
    def iree_src_dir(self):
        return self.config["iree.src_dir"]

    @property
    def iree_compile_exe(self):
        return Path(self.iree_install_dir) / "bin" / "iree-compile"

    @property
    def iree_c_embed_data_exe(self):
        return Path(self.iree_install_dir) / "bin" / "iree-c-embed-data"

    @property
    def iree_tflite_path(self):
        return Path(self.iree_src_dir) / "integrations" / "tensorflow" / "python_projects" / "iree_tflite"

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
        pythonpath = f"{self.iree_tflite_path}:{pythonpath}"
        print("pythonpath", pythonpath)
        env["PYTHONPATH"] = pythonpath
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
            model_path = self.model
            model_info = self.model_info
            translated = False
            if self.model_format != "mlir":
                translated = True
                mlirbc_path = out_dir / f"{self.identifier}.mlirbc"
                mlir_path = out_dir / f"{self.identifier}.mlir"
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
                    if major == 3 and minor > 1:
                        raise RuntimeError("TFLite (TOSA) importer unsupported for iree.version > 3.1")
                    python_args = ["-m", "iree.tools.tflite.scripts.iree_import_tflite", model_path, "-o", mlirbc_path]
                    needs_mlirbc2mlir = True
                elif self.model_format == "onnx":
                    python_args = [
                        "-m",
                        "iree.tools.onnx.scripts.iree_import_onnx",
                        model_path,
                        "-o",
                        mlir_path,
                        "--opset-version=17",
                    ]
                elif self.model_format == "saved_model":
                    python_args = [
                        "-m",
                        "iree.tools.tf.scripts.iree_import_tf",
                        model_path,
                        "-o",
                        mlir_path,
                        "--tf-import-type=savedmodel_v1",
                        "--tf-savedmodel-exported-names=predict",
                    ]
                else:
                    raise NotImplementedError(f"Unhandled format: {self.model_format}")
                utils.python(*python_args, live=self.print_outputs, env=self.prepare_environment(), cwd=temp_dir)
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
