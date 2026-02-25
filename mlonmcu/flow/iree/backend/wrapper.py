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
"""Wrapper generation utils for IREE backends."""
from typing import Optional

from mlonmcu.logging import get_logger
from mlonmcu.flow.tvm.backend.wrapper import getSizes
from .iree_utils import parse_iree_version


logger = get_logger()


def writeTensors(in_tensors, out_tensors):
    """Wrapper helper for generating tensors code."""
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


def getIOSetupCode(in_tensors, out_tensors, use_emitc: bool = False):
    """Wrapper util for generating I/O code."""
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


def getCopyOutputsCode(out_tensors):
    """Wrapper util for generating code to copy outputs."""
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


def generate_utils(identifier: str, use_emitc: bool, vmvx: bool, iree_version: Optional[str] = None):
    """Generate IREE wrapper code (utils)."""
    major, minor = parse_iree_version(iree_version)
    new = major > 4 or (major == 3 and minor >= 10)
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
      instance, """
        + ("IREE_VM_BYTECODE_MODULE_FLAG_NONE, " if new else "")
        + """module_data, iree_allocator_null(),
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
  return iree_vm_bytecode_module_create(instance,"""
        + ("IREE_VM_BYTECODE_MODULE_FLAG_NONE, " if new else "")
        + """ module_data,
                                        iree_allocator_null(),
                                        iree_allocator_system(), out_module);
}
"""
    )
    utils_ = utils_vmvx if vmvx else (utils_emitc if use_emitc else utils_bytecode)
    return utils_


def generate_sync(identifier: str, identifier2: str, vmvx: bool):
    """Generate IREE wrapper code (device_sync)."""
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
    sync = sync_vmvx if vmvx else sync_static
    return sync


def generate_wrapper(model_info, main_func_name: str, iree_version: Optional[str] = None):
    """Generate IREE wrapper code (source)."""
    inSizes = getSizes(model_info.in_tensors)
    outSizes = getSizes(model_info.out_tensors)
    numInputs = len(model_info.in_tensors)
    numOutputs = len(model_info.out_tensors)

    tensorBufs = writeTensors(model_info.in_tensors, model_info.out_tensors)

    # setupInputsOutputs = getIOSetupCode(model_info.in_tensors, model_info.out_tensors, use_emitc=use_emitc)
    setupInputsOutputs = getIOSetupCode(model_info.in_tensors, model_info.out_tensors, use_emitc=False)

    copyOutputs = getCopyOutputsCode(model_info.out_tensors)
    major, minor = parse_iree_version(iree_version)
    new = major > 4 or (major == 3 and minor >= 5)

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
  // printf("A\\n");
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
  // printf("B\\n");

  iree_hal_executable_loader_t* loader = NULL;
  IREE_RETURN_IF_ERROR(
      create_sample_device(iree_allocator_system(), &device, &loader),
      "create device");
  // printf("C\\n");

#if defined(BUILD_INLINE_HAL) || defined(BUILD_LOADER_HAL)
  // Create hal_inline_module
  iree_vm_module_t* hal_inline_module = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_inline_module_create(
      instance, IREE_HAL_INLINE_MODULE_FLAG_NONE,
      iree_hal_module_debug_sink_stdio(stderr),
      iree_hal_device_allocator(device), iree_allocator_system(),
      &hal_inline_module));
#endif
  // printf("D\\n");


  iree_vm_module_t *module = NULL;
  IREE_RETURN_IF_ERROR(create_module(instance, &module));
  // printf("E\\n");

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
      instance, """
        + ("iree_hal_module_device_policy_default(), " if new else "")
        + """/*device_count=*/1, &device, IREE_HAL_MODULE_FLAG_SYNCHRONOUS,
      iree_hal_module_debug_sink_stdio(stderr), iree_allocator_system(),
      &hal_module));

  iree_vm_module_t* modules[] = {hal_module, module};
#endif
  iree_hal_executable_loader_release(loader);
  // printf("F\\n");

  // Allocate a context that will hold the module state across invocations.
  // iree_vm_module_t *modules[] = {hal_module, module};
  IREE_RETURN_IF_ERROR(iree_vm_context_create_with_modules(
      instance, IREE_VM_CONTEXT_FLAG_NONE, IREE_ARRAYSIZE(modules), &modules[0],
      iree_allocator_system(), &context));
  // printf("G\\n");
  // iree_vm_module_release(hal_module);
#if defined(BUILD_INLINE_HAL) || defined(BUILD_LOADER_HAL)
  iree_vm_module_release(hal_inline_module);
#else
  iree_vm_module_release(hal_module);
#endif
  // printf("H\\n");

#if defined(BUILD_LOADER_HAL)
  iree_vm_module_release(hal_loader_module);
#endif
  iree_vm_module_release(module);
  // printf("I\\n");

  // Lookup the entry point function.
  // Note that we use the synchronous variant which operates on pure type/shape
  // erased buffers.
  const char kMainFunctionName[] = \"module."""
        + main_func_name
        + """\";
  IREE_RETURN_IF_ERROR(iree_vm_context_resolve_function(
      context, iree_make_cstring_view(kMainFunctionName), &main_function));
  // printf("J\\n");

  // Allocate buffers in device-local memory so that if the device has an
  // independent address space they live on the fast side of the fence.
  """
        + setupInputsOutputs
        + """
  // printf("KLMNOP\\n");

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
  // printf("Q\\n");

  // Get the result buffers from the invocation.
  """
        + copyOutputs
        + """
  // printf("S\\n");
  // for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(results); ++i) {
  //   if (results[i] != 8) {
  //     return iree_make_status(IREE_STATUS_UNKNOWN, "result mismatches");
  //   }
  // }

  // Print statistics (no-op if statistics are not enabled).
  iree_hal_allocator_statistics_fprint(stdout,
                                       iree_hal_device_allocator(device));
  // printf("T\\n");

  iree_vm_list_release(inputs_);
  iree_vm_list_release(outputs_);
  iree_hal_device_release(device);
  iree_vm_context_release(context);
  iree_vm_instance_release(instance);
  // printf("U\\n");
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
    prolog = ""
    wrapper = prolog + wrapper_main + epilog
    return wrapper


def generate_header():
    """Generate IREE wrapper code (header)."""
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
    return header


def generate_iree_wrapper(
    model_info,
    identifier: str,
    use_emitc: bool = False,
    vmvx: bool = False,
    translated: bool = False,
    iree_version: Optional[str] = None,
):
    """Generate IREE wrapper codes (source, header, device_sync, utils)."""

    logger.debug("Generating IREE wrapper...")
    main_func_name = model_info.main_func_name
    assert main_func_name is not None
    identifier2 = f"{identifier}_linked" if translated else f"{main_func_name}_dispatch_0"

    wrapper = generate_wrapper(model_info, main_func_name, iree_version=iree_version)
    header = generate_header()
    sync = generate_sync(identifier, identifier2, vmvx)
    utils_ = generate_utils(identifier, use_emitc, vmvx, iree_version=iree_version)
    return wrapper, header, sync, utils_
