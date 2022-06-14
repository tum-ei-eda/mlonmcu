#[[
Copyright (c) 2022 TUM Department of Electrical and Computer Engineering.

This file is part of MLonMCU.
See https://github.com/tum-ei-eda/mlonmcu.git for further info.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
]]
cmake_minimum_required(VERSION 3.5)

# set(TF_SRC "${CONFIG_TF_DIR}" CACHE PATH "Tensorflow (Lite Micro) Source Directory")
set(TF_SRC "${CONFIG_TF_DIR}")

IF (NOT ${TF_SRC} STREQUAL "")

set(TFL_SRC "${TF_SRC}/tensorflow/lite/")
set(TFLM_SRC "${TFL_SRC}/micro")
set(TFLD_SRC "${TFLM_SRC}/tools/make/downloads")

set(srcs_micro
    ${TFLM_SRC}/micro_error_reporter.cc
    ${TFLM_SRC}/debug_log.cc
    ${TFLM_SRC}/micro_string.cc
    ${TF_SRC}/tensorflow/lite/core/api/error_reporter.cc
    ${TFLM_SRC}/kernels/kernel_util.cc
    ${TFLM_SRC}/all_ops_resolver.cc
    ${TFLM_SRC}/micro_utils.cc
    ${TFL_SRC}/kernels/internal/quantization_util.cc
    ${TFL_SRC}/kernels/kernel_util.cc
    ${TFL_SRC}/c/common.cc
    ${TFLM_SRC}/micro_interpreter.cc
    ${TFLM_SRC}/micro_allocator.cc
    ${TFLM_SRC}/micro_allocation_info.cc
    ${TFLM_SRC}/arena_allocator/simple_memory_allocator.cc
    ${TFLM_SRC}/memory_helpers.cc
    ${TFLM_SRC}/memory_planner/greedy_memory_planner.cc
    ${TFL_SRC}/core/api/tensor_utils.cc
    ${TFL_SRC}/core/api/flatbuffer_conversions.cc
    ${TFL_SRC}/core/api/op_resolver.cc
    ${TFL_SRC}/schema/schema_utils.cc
    ${TFLM_SRC}/micro_context.cc
    ${TFLM_SRC}/micro_graph.cc
    ${TFLM_SRC}/micro_resource_variable.cc
    ${TFLM_SRC}/flatbuffer_utils.cc
    )

set(srcs_kernels
    ${TFLM_SRC}/kernels/softmax.cc
    ${TFLM_SRC}/kernels/fully_connected.cc
    ${TFLM_SRC}/kernels/pooling.cc
    ${TFLM_SRC}/kernels/add.cc
    ${TFLM_SRC}/kernels/mul.cc
    ${TFLM_SRC}/kernels/conv.cc
    ${TFLM_SRC}/kernels/depthwise_conv.cc
    ${TFLM_SRC}/kernels/softmax.cc
    ${TFLM_SRC}/kernels/fully_connected.cc
    ${TFLM_SRC}/kernels/pooling.cc
    ${TFLM_SRC}/kernels/add.cc
    ${TFLM_SRC}/kernels/mul.cc
    ${TFLM_SRC}/kernels/conv.cc
    ${TFLM_SRC}/kernels/depthwise_conv.cc
    ${TFLM_SRC}/kernels/logical.cc
    ${TFLM_SRC}/kernels/logistic.cc
    ${TFLM_SRC}/kernels/svdf.cc
    ${TFLM_SRC}/kernels/concatenation.cc
    ${TFLM_SRC}/kernels/ceil.cc
    ${TFLM_SRC}/kernels/floor.cc
    ${TFLM_SRC}/kernels/prelu.cc
    ${TFLM_SRC}/kernels/neg.cc
    ${TFLM_SRC}/kernels/elementwise.cc
    ${TFLM_SRC}/kernels/maximum_minimum.cc
    ${TFLM_SRC}/kernels/arg_min_max.cc
    ${TFLM_SRC}/kernels/reshape.cc
    ${TFLM_SRC}/kernels/comparisons.cc
    ${TFLM_SRC}/kernels/round.cc
    ${TFLM_SRC}/kernels/strided_slice.cc
    ${TFLM_SRC}/kernels/pack.cc
    ${TFLM_SRC}/kernels/pad.cc
    ${TFLM_SRC}/kernels/split.cc
    ${TFLM_SRC}/kernels/unpack.cc
    ${TFLM_SRC}/kernels/quantize.cc
    ${TFLM_SRC}/kernels/activations.cc
    ${TFLM_SRC}/kernels/dequantize.cc
    ${TFLM_SRC}/kernels/reduce.cc
    ${TFLM_SRC}/kernels/sub.cc
    ${TFLM_SRC}/kernels/resize_nearest_neighbor.cc
    ${TFLM_SRC}/kernels/l2norm.cc
    ${TFLM_SRC}/kernels/circular_buffer.cc
    ${TFLM_SRC}/kernels/ethosu.cc
    ${TFLM_SRC}/kernels/tanh.cc
    ${TFLM_SRC}/kernels/conv_common.cc
    ${TFLM_SRC}/kernels/depthwise_conv_common.cc
    ${TFLM_SRC}/kernels/fully_connected_common.cc
    ${TFLM_SRC}/kernels/quantize_common.cc
    ${TFLM_SRC}/kernels/dequantize_common.cc
    ${TFLM_SRC}/kernels/softmax_common.cc
    ${TFLM_SRC}/kernels/svdf_common.cc
    ${TFLM_SRC}/kernels/pooling_common.cc
    ${TFLM_SRC}/kernels/add_common.cc
    ${TFLM_SRC}/kernels/detection_postprocess.cc
)

set(LIB_SRCS
          "${srcs_micro}"
          "${srcs_kernels}"
)


idf_component_register(
    SRCS "${LIB_SRCS}"
            INCLUDE_DIRS "." "${TF_SRC}" "${TFLD_SRC}/gemmlowp"
                         "${TFLD_SRC}/flatbuffers/include"
                         "${TFLD_SRC}/ruy"
                         "${TFLD_SRC}/kissfft"
            )
            # REQUIRES "esp-nn")

# Reduce the level of paranoia to be able to compile TF sources
target_compile_options(${COMPONENT_LIB} PRIVATE
  -Wno-maybe-uninitialized
  -Wno-missing-field-initializers
  -DESP_NN # enables ESP-NN optimizations by Espressif
  -Wno-type-limits)

target_compile_options(${COMPONENT_LIB} PRIVATE -fno-unwind-tables -ffunction-sections -fdata-sections -fmessage-length=0 -DTF_LITE_STATIC_MEMORY -DTF_LITE_DISABLE_X86_NEON -O3 -Wsign-compare -Wdouble-promotion -Wshadow -Wunused-variable -Wmissing-field-initializers -Wunused-function -Wswitch -Wvla -Wall -Wextra -Wstrict-aliasing -Wno-unused-parameter -Wno-nonnull)
target_compile_options(${COMPONENT_LIB} PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -std=c++11 -fno-rtti -fno-exceptions -fno-threadsafe-statics -fno-unwind-tables -ffunction-sections -fdata-sections -fmessage-length=0 -DTF_LITE_STATIC_MEMORY -DTF_LITE_DISABLE_X86_NEON -O3 -Werror -Wsign-compare -Wdouble-promotion -Wshadow -Wunused-variable -Wmissing-field-initializers -Wunused-function -Wswitch -Wvla -Wall -Wextra -Wstrict-aliasing -Wno-unused-parameter -Wno-return-type -Wno-strict-aliasing -std=gnu++14 >)
target_compile_options(${COMPONENT_LIB} INTERFACE $<$<IN_LIST:-DTF_LITE_STATIC_MEMORY,$<TARGET_PROPERTY:${COMPONENT_LIB},COMPILE_OPTIONS>>:-DTF_LITE_STATIC_MEMORY>)
target_link_libraries(${COMPONENT_LIB} PRIVATE -lm)

ELSE()
    idf_component_register()
ENDIF()
