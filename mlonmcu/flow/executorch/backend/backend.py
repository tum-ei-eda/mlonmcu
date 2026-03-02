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
from typing import Tuple

from mlonmcu.flow.backend import Backend
from mlonmcu.setup import utils
from mlonmcu.config import str2bool
from mlonmcu.logging import get_logger
from mlonmcu.models.model_info import (
    get_model_info,
)
from mlonmcu.models.torch_models.torch_utils import load_torch_model
from mlonmcu.target.metrics import Metrics
from mlonmcu.artifact import Artifact, ArtifactFormat

from mlonmcu.models.model import ModelFormats

logger = get_logger()


def fill(template, **kwargs):
    return string.Template(template).substitute(**kwargs)


def generate_executorch_wrapper(
    model_info,
    identifier: str,
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
int {prefix}_deinit();
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
        # TODO: replace
        out = """// Disclaimer: This file is heavily inspired by
// https://github.com/pytorch/executorch/tree/main/examples/arm/executor_runner
#include <errno.h>
#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/runner_util/inputs.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>
#include <stdio.h>
#include <unistd.h>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

// ---
using executorch::runtime::MemoryAllocator;

#pragma once

// Setup our own allocator that can show some extra stuff like used and free
// memory info
class MlonmcuMemoryAllocator : public executorch::runtime::MemoryAllocator {
 public:
  MlonmcuMemoryAllocator(uint32_t size, uint8_t* base_address);

  void* allocate(size_t size, size_t alignment = kDefaultAlignment) override;

  // Returns the used size of the allocator's memory buffer.
  size_t used_size() const;

  // Returns the free size of the allocator's memory buffer.
  size_t free_size() const;
  void reset();

 private:
  size_t used_;
};

#if defined(EXECUTORCH_ENABLE_ADDRESS_SANITIZER)
extern "C" {
void __asan_poison_memory_region(void* addr, size_t size);
void __asan_unpoison_memory_region(void* addr, size_t size);
}

static void asan_poison_buffer(uint8_t* base, size_t size) {
  if (base != nullptr && size > 0) {
    __asan_poison_memory_region(base, size);
  }
}

static void asan_unpoison_buffer(void* base, size_t size) {
  if (base != nullptr && size > 0) {
    __asan_unpoison_memory_region(base, size);
  }
}
#endif

MlonmcuMemoryAllocator::MlonmcuMemoryAllocator(uint32_t size, uint8_t* base_address)
    : MemoryAllocator(size, base_address), used_(0) {
#if defined(EXECUTORCH_ENABLE_ADDRESS_SANITIZER)
  asan_poison_buffer(base_address, size);
#endif
}

void* MlonmcuMemoryAllocator::allocate(size_t size, size_t alignment) {
  void* ret = executorch::runtime::MemoryAllocator::allocate(size, alignment);
  if (ret != nullptr) {
#if defined(EXECUTORCH_ENABLE_ADDRESS_SANITIZER)
    asan_unpoison_buffer(ret, size);
#endif
    // Align with the same code as in MemoryAllocator::allocate() to keep
    // used_ "in sync" As alignment is expected to be power of 2 (checked by
    // MemoryAllocator::allocate()) we can check it the lower bits
    // (same as alignment - 1) is zero or not.
    if ((size & (alignment - 1)) == 0) {
      // Already aligned.
      used_ += size;
    } else {
      used_ = (used_ | (alignment - 1)) + 1 + size;
    }
  }
  return ret;
}

size_t MlonmcuMemoryAllocator::used_size() const {
  return used_;
}

size_t MlonmcuMemoryAllocator::free_size() const {
  return executorch::runtime::MemoryAllocator::size() - used_;
}

void MlonmcuMemoryAllocator::reset() {
  executorch::runtime::MemoryAllocator::reset();
  used_ = 0;
#if defined(EXECUTORCH_ENABLE_ADDRESS_SANITIZER)
  asan_poison_buffer(base_address(), size());
#endif
}

// ---

#include "target.h"
#include "printing.h"
#include "exit.h"

#include "${prefix}_pte.h"
#include "executorch_wrapper.h"

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::extension::BufferDataLoader;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorch::runtime::Tag;
using executorch::runtime::TensorInfo;
using executorch::runtime::toString;
/**
 * The method_allocation_pool should be large enough to fit the setup, input
 * used and other data used like the planned memory pool (e.g. memory-planned
 * buffers to use for mutable tensor data) In this example we run on a
 * Corstone-3xx FVP so we can use a lot of memory to be able to run and test
 * large models if you run on HW this should be lowered to fit into your
 * availible memory.
 */
#if !defined(ET_MLONMCU_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE)
#define ET_MLONMCU_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE (${method_pool_size})
#endif
const size_t method_allocation_pool_size =
    ET_MLONMCU_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE;
unsigned char __attribute__((
    section("input_data_sec"),
    aligned(16))) method_allocation_pool[method_allocation_pool_size];


/**
 * The temp_allocation_pool is used for allocating temporary data during kernel
 * or delegate execution. This will be reset after each kernel or delegate call.
 * Currently a MemoryAllocator is used but a PlatformMemoryAllocator is probably
 * a better fit.
 *
 * The Corstone-300/Corstone-320 platforms have 2MB/4MB of SRAM respectively.
 * For Shared_Sram, ET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE is
 * 2MB and the linker script places the .bss.tensor_arena symbol in the SRAM.
 * For Dedicated_Sram, the .bss.tensor_arena symbol is placed in the DDR in the
 * linker script. Hence, we allocate 128MB in DDR and 384KB in the SRAM
 * (.bss.ethosu_scratch is placed in the SRAM). The examples/arm/CMakeLists.txt
 * contains the logic for the sizes of
 * ET_MLONMCU_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE and
 * ET_MLONMCU_BAREMETAL_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE
 */

// TODO
#if !defined(ET_MLONMCU_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE)
#define ET_MLONMCU_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE (${temp_pool_size})
#endif

const size_t temp_allocation_pool_size =
    ET_MLONMCU_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE;
unsigned char __attribute__((
    section(".bss.tensor_arena"),
    aligned(16))) temp_allocation_pool[temp_allocation_pool_size];
#if defined(ET_MLONMCU_BAREMETAL_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE)
extern "C" {
size_t ethosu_fast_scratch_size =
    ET_MLONMCU_BAREMETAL_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE;
unsigned char __attribute__((section(".bss.ethosu_scratch"), aligned(16)))
dedicated_sram[ET_MLONMCU_BAREMETAL_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE];
unsigned char* ethosu_fast_scratch = dedicated_sram;
}
#endif

void et_pal_init(void) {
}

/**
 * Implementation of the et_pal_<funcs>()
 *
 * This functions are hardware adaption type of functions for things like
 * time/logging/memory allocation that could call your RTOS or need to to
 * be implemnted in some way.
 */

ET_NORETURN void et_pal_abort(void) {
  mlonmcu_exit(-1);
}

et_timestamp_t et_pal_current_ticks(void) {
  return target_cycles();  // TODO: time!?
  // return 0;
}

et_tick_ratio_t et_pal_ticks_to_ns_multiplier(void) {
  // Since we don't know the CPU freq for your target and justs cycles in the
  // FVP for et_pal_current_ticks() we return a conversion ratio of 1
  return {1, 1};
}

/**
 * Emit a log message via platform output (serial port, console, etc).
 */
void et_pal_emit_log_message(
    ET_UNUSED et_timestamp_t timestamp,
    et_pal_log_level_t level,
    const char* filename,
    ET_UNUSED const char* function,
    size_t line,
    const char* message,
    ET_UNUSED size_t length) {
  fprintf( // TODO
      stderr,
      "%c [executorch:%s:%lu %s()] %s\\n",
      level,
      filename,
      static_cast<unsigned long>(line),
      function,
      message);
}

/**
 * Dynamic memory allocators intended to be used by temp_allocator
 * to implement malloc()/free() type of allocations.
 * Currenyly not used.
 */

void* et_pal_allocate(ET_UNUSED size_t size) {
  return nullptr;
}

void et_pal_free(ET_UNUSED void* ptr) {}

namespace {

/// Lightweight heapless container that constructs and stores a T in-place.
/// Useful when you want to avoid heap allocations but need to delay
/// construction.
template <typename T>
class Box {
 public:
  Box() = default;

  ~Box() {
    if (has_value) {
      ptr()->~T();
    }
  }

  Box(const Box&) = delete;
  Box& operator=(const Box&) = delete;

  /// Destructs the already contained object if it's present and initialize a
  /// new contained object while forwarding its constructor arguments.
  template <typename... Args>
  void reset(Args&&... args) {
    if (has_value) {
      // Destroy the already contained object.
      reinterpret_cast<T*>(mem)->~T();
    }
    // Init the new object.
    new (mem) T(std::forward<Args>(args)...);
    has_value = true;
  }

  /// Returns a reference to the contained object.
  T& value() {
    return *ptr();
  }

  /// Returns a const reference to the contained object.
  const T& value() const {
    return *ptr();
  }

  T* operator->() {
    return ptr();
  }

  const T* operator->() const {
    return ptr();
  }

 private:
  alignas(T) uint8_t mem[sizeof(T)];
  bool has_value = false;

  T* ptr() {
    return reinterpret_cast<T*>(mem);
  }

  const T* ptr() const {
    return reinterpret_cast<const T*>(mem);
  }
};

template <typename ValueType>
void fill_tensor_with_default_value(Tensor& tensor) {
  ValueType fill_value{};
  if constexpr (std::is_same_v<ValueType, bool>) {
    fill_value = true;
  } else {
    fill_value = ValueType(1);
  }

  ValueType* data_ptr = tensor.mutable_data_ptr<ValueType>();
  std::fill(data_ptr, data_ptr + tensor.numel(), fill_value);
}

Error prepare_input_tensors(
    Method& method,
    MemoryAllocator& allocator,
    const std::vector<std::pair<char*, size_t>>& input_buffers) {
  MethodMeta method_meta = method.method_meta();
  size_t num_inputs = method_meta.num_inputs();
  size_t num_outputs = method_meta.num_outputs();
  size_t num_allocated = 0;

  EValue* input_evalues = allocator.allocateList<EValue>(num_inputs);
  ET_CHECK_OR_RETURN_ERROR(
      input_evalues != nullptr,
      MemoryAllocationFailed,
      "Could not allocate memory for input evalues.");

  Error err = method.get_inputs(input_evalues, num_inputs);
  ET_CHECK_OK_OR_RETURN_ERROR(err);

  for (size_t i = 0; i < num_inputs; i++) {
    auto tag = method_meta.input_tag(i);
    ET_CHECK_OK_OR_RETURN_ERROR(tag.error());

    if (tag.get() != Tag::Tensor) {
      ET_LOG(
          Debug,
          "Skipping non-tensor input %lu",
          static_cast<unsigned long>(i));
      continue;
    }
    Result<TensorInfo> tensor_meta = method_meta.input_tensor_meta(i);
    ET_CHECK_OK_OR_RETURN_ERROR(tensor_meta.error());

    err = Error::Ok;
    if (input_buffers.size() > 0) {
      auto [buffer, buffer_size] = input_buffers.at(i);
      if (buffer_size != tensor_meta->nbytes()) {
        ET_LOG(
            Error,
            "input size (%d) and tensor size (%d) mismatch!",
            buffer_size,
            tensor_meta->nbytes());
        err = Error::InvalidArgument;
      } else if (input_evalues[i].isTensor()) {
        // Copy the data from the input buffer to the tensor
        Tensor& tensor = input_evalues[i].toTensor();
        std::memcpy(tensor.mutable_data_ptr<int8_t>(), buffer, buffer_size);
      }
    }

    // If input_buffers.size <= 0, we don't have any input, fill it with 1's.
    if (input_buffers.size() <= 0) {
      if (input_evalues[i].isTensor()) {
        Tensor& tensor = input_evalues[i].toTensor();
        switch (tensor.scalar_type()) {
#define HANDLE_SCALAR_TYPE(cpp_type, scalar_name)     \\
  case ScalarType::scalar_name:                       \\
    fill_tensor_with_default_value<cpp_type>(tensor); \\
    break;
          ET_FORALL_SCALAR_TYPES(HANDLE_SCALAR_TYPE)
#undef HANDLE_SCALAR_TYPE
          default:
            ET_LOG(
                Error,
                "Unhandled ScalarType %s",
                toString(tensor.scalar_type()));
            err = Error::InvalidArgument;
            break;
        }
      } else {
        printf("Input[%d]: Not Tensor\\n", i);
      }
    }
  }

  return err;
}

/// Holds all state needed for setup and run phases
struct RunnerContext {
  RunnerContext() = default;
  RunnerContext(const RunnerContext& ctx) = delete;
  RunnerContext& operator=(const RunnerContext& ctx) = delete;

  const char* method_name = nullptr;
  size_t planned_buffer_memsize = 0;
  size_t method_loaded_memsize = 0;
  size_t executor_membase = 0;
  size_t program_data_len = 0;
  size_t input_memsize = 0;
  size_t pte_size = 0;
  bool bundle_io = false;
  Box<BufferDataLoader> loader;
  Box<Program> program;
  Box<MlonmcuMemoryAllocator> method_allocator;
  Box<MlonmcuMemoryAllocator> temp_allocator;
  std::vector<Span<uint8_t>> planned_spans;
  Box<HierarchicalAllocator> planned_memory;
  Box<MemoryManager> memory_manager;
  Box<Result<Method>> method;
};

void runner_init(
    RunnerContext& ctx,
    std::vector<std::pair<char*, size_t>> input_buffers,
    size_t pte_size) {
  // Find the offset to the embedded Program.
  const void* program_data = model_pte;
  ctx.program_data_len = pte_size;
  ctx.pte_size = pte_size;

  ctx.loader.reset(program_data, ctx.program_data_len);
  auto& loader = ctx.loader.value();
  ET_LOG(
      Info,
      "PTE Model data loaded. Size: %lu bytes.",
      static_cast<unsigned long>(ctx.program_data_len));

  // Parse the program file. This is immutable, and can also be reused
  // between multiple execution invocations across multiple threads.
  Result<Program> program_result = Program::load(&loader);
  ET_CHECK_MSG(
      program_result.ok(),
      "Program loading failed @ %p: 0x%" PRIx32,
      program_data,
      program_result.error());
  ctx.program.reset(std::move(program_result.get()));
  Program& program = ctx.program.value();

  ET_LOG(
      Info,
      "Model buffer loaded, has %lu methods",
      static_cast<unsigned long>(program.num_methods()));

  {
    const auto method_name_result = program.get_method_name(0);
    ET_CHECK_MSG(method_name_result.ok(), "Program has no methods");
    ctx.method_name = *method_name_result;
  }
  ET_LOG(Info, "Running method %s", ctx.method_name);

  Result<MethodMeta> method_meta = program.method_meta(ctx.method_name);
  if (!method_meta.ok()) {
    ET_LOG(
        Info,
        "Failed to get method_meta for %s: 0x%x",
        ctx.method_name,
        (unsigned int)method_meta.error());
  }

  ET_LOG(
      Info,
      "Setup Method allocator pool. Size: %lu bytes.",
      static_cast<unsigned long>(method_allocation_pool_size));

  ctx.method_allocator.reset(
      method_allocation_pool_size, method_allocation_pool);

  ctx.planned_spans.clear();
  size_t num_memory_planned_buffers = method_meta->num_memory_planned_buffers();
  ctx.planned_spans.reserve(num_memory_planned_buffers);
  size_t planned_buffer_membase = ctx.method_allocator->used_size();

  for (size_t id = 0; id < num_memory_planned_buffers; ++id) {
    size_t buffer_size =
        static_cast<size_t>(method_meta->memory_planned_buffer_size(id).get());
    ET_LOG(
        Info,
        "Setting up planned buffer %lu, size %lu.",
        static_cast<unsigned long>(id),
        static_cast<unsigned long>(buffer_size));

    /* Move to it's own allocator when MemoryPlanner is in place. */
    /* Ethos-U driver requires 16 bit alignment. */
    uint8_t* buffer = reinterpret_cast<uint8_t*>(
        ctx.method_allocator->allocate(buffer_size, 16UL));
    ET_CHECK_MSG(
        buffer != nullptr,
        "Could not allocate memory for memory planned buffer size %lu",
        static_cast<unsigned long>(buffer_size));
    ctx.planned_spans.push_back({buffer, buffer_size});
  }

  ctx.planned_buffer_memsize =
      ctx.method_allocator->used_size() - planned_buffer_membase;

  Span<Span<uint8_t>> planned_memory_span;
  if (!ctx.planned_spans.empty()) {
    planned_memory_span =
        Span<Span<uint8_t>>(ctx.planned_spans.data(), ctx.planned_spans.size());
  }
  ctx.planned_memory.reset(planned_memory_span);

  ctx.temp_allocator.reset(temp_allocation_pool_size, temp_allocation_pool);

  ctx.memory_manager.reset(
      &ctx.method_allocator.value(),
      &ctx.planned_memory.value(),
      &ctx.temp_allocator.value());

  size_t method_loaded_membase = ctx.method_allocator->used_size();

  executorch::runtime::EventTracer* event_tracer_ptr = nullptr;

  ctx.method.reset(program.load_method(
      ctx.method_name, &ctx.memory_manager.value(), event_tracer_ptr));

  if (!ctx.method->ok()) {
    ET_LOG(
        Info,
        "Loading of method %s failed with status 0x%" PRIx32,
        ctx.method_name,
        ctx.method->error());
  }
  ctx.method_loaded_memsize =
      ctx.method_allocator->used_size() - method_loaded_membase;
  ET_LOG(Info, "Method '%s' loaded.", ctx.method_name);

  ET_LOG(Info, "Preparing inputs...");
  size_t input_membase = ctx.method_allocator->used_size();

  {
    Error status = ::prepare_input_tensors(
        *ctx.method.value(), ctx.method_allocator.value(), input_buffers);
    ET_CHECK_MSG(
        status == Error::Ok, "Failed to prepare inputs 0x%" PRIx32, status);
  }
  ctx.input_memsize = ctx.method_allocator->used_size() - input_membase;
  ctx.executor_membase = ctx.method_allocator->used_size();

  ET_LOG(Info, "Input prepared.");
}

void log_mem_status(RunnerContext& ctx) {
  size_t executor_memsize =
      ctx.method_allocator->used_size() - ctx.executor_membase;

  ET_LOG(
      Info,
      "model_pte_program_size:     %lu bytes.",
      static_cast<unsigned long>(ctx.program_data_len));
  ET_LOG(
      Info,
      "model_pte_loaded_size:      %lu bytes.",
      static_cast<unsigned long>(ctx.pte_size));

}

bool verify_result(RunnerContext& ctx, const void* model_pte) {
  bool model_ok = false;
  (void)ctx;
  (void)model_pte;
  // No checking done, assume true
  model_ok = true;
  return model_ok;
}

bool run_model(RunnerContext& ctx, const void* model_pte) {
  Error status;
  status = ctx.method.value()->execute();
  // Reset the temporary allocator holding the scratch buffer between
  // inferences. We want to reuse the temp_allocator between inferences of the
  // same Ethos-U custom delegate, not allocate memory with every new
  // inference.
  ctx.temp_allocator.reset(temp_allocation_pool_size, temp_allocation_pool);

  ET_CHECK_MSG(
      status == Error::Ok,
      "Execution of method %s failed with status 0x%" PRIx32,
      ctx.method_name,
      status);

  // ET_LOG(Info, "%d inferences finished", num_inferences);
  bool model_ok = verify_result(ctx, model_pte);
  // ET_LOG(Info, "Model run: %d", model_ok);

  return !model_ok;
}

} // namespace


// ---

static RunnerContext ctx;

int ${prefix}_init()
{
    executorch::runtime::runtime_init();
    std::vector<std::pair<char*, size_t>> input_buffers;
    size_t pte_size = sizeof(model_pte);
    // Byte 4-7 is usually a nice magic number that could be good to print to make
    // sure it's OK ETxx for PTE and BPxx for bundled pte where xx is a number.
    ET_LOG(
        Info,
        "PTE @ %p [----%c%c%c%c]",
        model_pte,
        model_pte[4],
        model_pte[5],
        model_pte[6],
        model_pte[7]);

    runner_init(ctx, input_buffers, pte_size);

    return 0;
}

int ${prefix}_deinit()
{
    log_mem_status(ctx);  // TODO: optional
    // write_etdump(ctx);  // TODO: optional

    ET_LOG(Info, "Program complete, exiting.");
    ET_LOG(Info, "\\04");
    return 0;
}

char buf;

void* ${prefix}_input_ptr(int index)
{
    std::vector<EValue> inputs(ctx.method.value()->inputs_size());
    Error status =
        ctx.method.value()->get_inputs(inputs.data(), inputs.size());
    ET_CHECK(status == Error::Ok);

    ET_CHECK(inputs[index].isTensor());
    Tensor tensor = inputs[index].toTensor();
    return (void*)tensor.const_data_ptr<char>();
}

size_t ${prefix}_input_size(int index)
{
    // std::vector<EValue> inputs(ctx.method.value()->inputs_size());
    // Error status =
    //     ctx.method.value()->get_inputs(inputs.data(), inputs.size());
    // ET_CHECK(status == Error::Ok);

    // ET_CHECK(inputs[index].isTensor());
    // Tensor tensor = inputs[index].toTensor();
    // return tensor.nbytes();
    MethodMeta method_meta = ctx.method.value()->method_meta();
    Result<TensorInfo> tensor_meta = method_meta.input_tensor_meta(index);
    ET_CHECK(tensor_meta.error() == Error::Ok);

    return tensor_meta->nbytes();
}

size_t ${prefix}_inputs()
{
    MethodMeta method_meta = ctx.method.value()->method_meta();
    size_t num_inputs = method_meta.num_inputs();

    return num_inputs;
}

void* ${prefix}_output_ptr(int index)
{
    std::vector<EValue> outputs(ctx.method.value()->outputs_size());
    Error status =
        ctx.method.value()->get_outputs(outputs.data(), outputs.size());
    ET_CHECK(status == Error::Ok);

    ET_CHECK(outputs[index].isTensor());
    Tensor tensor = outputs[index].toTensor();
    return (void*)tensor.const_data_ptr<char>();
}

size_t ${prefix}_output_size(int index)
{
    // std::vector<EValue> outputs(ctx.method.value()->outputs_size());
    // Error status =
    //     ctx.method.value()->get_outputs(outputs.data(), outputs.size());
    // ET_CHECK(status == Error::Ok);

    // ET_CHECK(outputs[index].isTensor());
    // Tensor tensor = outputs[index].toTensor();
    // return tensor.nbytes();
    MethodMeta method_meta = ctx.method.value()->method_meta();
    Result<TensorInfo> tensor_meta = method_meta.output_tensor_meta(index);
    ET_CHECK(tensor_meta.error() == Error::Ok);

    return tensor_meta->nbytes();
}

size_t ${prefix}_outputs()
{
    MethodMeta method_meta = ctx.method.value()->method_meta();
    size_t num_outputs = method_meta.num_outputs();

    return num_outputs;
}

int ${prefix}_invoke()
{
    bool model_ok = run_model(ctx, model_pte);

    std::vector<EValue> outputs(ctx.method.value()->outputs_size());
    Error status = ctx.method.value()->get_outputs(outputs.data(), outputs.size());

    return model_ok;  // TODO
}
"""
        out = fill(
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
        self.supported_formats = [
            ModelFormats.PTE,
            ModelFormats.TORCH_PICKLE,
            ModelFormats.TORCH_PYTHON,
            ModelFormats.TORCH_EXPORTED,
        ]
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
        args = [self.pte_to_header_exe, "--pte", pte_file, "--outdir", outdir, "--outfile", outfile]
        env = self.prepare_environment()
        return utils.python(*args, live=self.print_outputs, env=env, cwd=cwd)

    def load_model(
        self, model, input_shapes=None, output_shapes=None, input_types=None, output_types=None, params_path=None
    ):
        assert params_path is None
        self.model = model
        self.model_format, self.model_info = get_model_info(model, backend_name=self.name)

    def generate(self) -> Tuple[dict, dict]:
        out = ""
        artifacts = []
        metrics = Metrics()
        assert self.model is not None
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            # model_path = self.model
            # model_info = self.model_info
            pte_file = out_dir / f"{self.identifier}.pte"
            if self.model_format == "pte":
                utils.copy(self.model, pte_file)
            elif self.model_format == "torch":

                from torch.export import ExportedProgram
                from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower
                from executorch.extension.export_util.utils import save_pte_program

                _, exported_program, _ = load_torch_model(self.model)

                def convert_torch_to_edge(exported_program: ExportedProgram, quantize: bool = False):
                    if quantize:
                        raise NotImplementedError("quantize")
                    compile_config = EdgeCompileConfig(_check_ir_validity=False)
                    edge = to_edge_transform_and_lower(exported_program, compile_config=compile_config)
                    return edge

                edge = convert_torch_to_edge(exported_program, quantize=False)

                exec_prog = edge.to_executorch(config=ExecutorchBackendConfig(extract_delegate_segments=False))

                save_pte_program(exec_prog, str(pte_file))

                with open(pte_file, "rb") as f:
                    model_raw = f.read()
                artifacts.append(
                    Artifact(
                        pte_file.name,
                        raw=model_raw,
                        fmt=ArtifactFormat.BIN,
                    )
                )
            else:
                raise RuntimeError(f"Unsupported format: {self.model_format}")
            pte_header_file = out_dir / f"{self.identifier}_pte.h"
            out += self.generate_pte_header(pte_file, pte_header_file)
            with open(pte_header_file, "r") as f:
                pte_header_content = f.read()
            artifacts.append(
                Artifact(
                    pte_header_file.name,
                    content=pte_header_content,
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
                    content=header_content,
                    fmt=ArtifactFormat.SOURCE,
                )
            )
            stdout_artifact = Artifact("executorch_out.log", content=out, fmt=ArtifactFormat.TEXT)
            artifacts.append(stdout_artifact)
        return {"default": artifacts}, {"default": metrics}

    def get_platform_defs(self, platform):
        ret = super().get_platform_defs(platform)
        if self.model:
            if self.model_format == "torch":
                model_path = Path(self.model)
                # model_name = model_path.stem
                pte_file = model_path.parent / f"{self.identifier}.pte"  # Needs exported artifact!
            else:
                assert self.model_format == "pte"
                pte_file = self.model
            ret["EXECUTORCH_PTE_FILE_PATH"] = pte_file
        return ret
