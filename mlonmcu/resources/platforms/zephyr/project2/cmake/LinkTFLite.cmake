SET(TF_DIR
    "/data/work/code/tensorflow"
    CACHE PATH "TensorFlow source directory"
)

SET(TFL_SRC ${CONFIG_TF_DIR}/tensorflow/lite)
SET(TFLM_SRC ${TFL_SRC}/micro)
SET(TFLD_SRC ${TFLM_SRC}/tools/make/downloads)

SET(TFLM_EXTRA_KERNEL_LIBS "")
SET(TFLM_EXTRA_KERNEL_INCS "")

IF(CONFIG_TFLM_OPTIMIZED_KERNEL)
    # Suboptimal but we do not want to hardcode every kernel which should be replaced...
    FILE(GLOB TFLM_EXTRA_KERNEL_SRCS ${TFLM_SRC}/kernels/${CONFIG_TFLM_OPTIMIZED_KERNEL}/*.cc)
    # LIST(APPEND TFLM_EXTRA_KERNEL_INCS ${TFLM_SRC}/kernels/${TFLM_OPTIMIZED_KERNEL}/)
    STRING(TOUPPER "${CONFIG_TFLM_OPTIMIZED_KERNEL}" TFLM_OPTIMIZED_KERNEL_UPPER)
ENDIF()

IF(CONFIG_TFLM_OPTIMIZED_KERNEL_LIB)
    LIST(APPEND TFLM_EXTRA_KERNEL_LIBS ${CONFIG_TFLM_OPTIMIZED_KERNEL_LIB})
ENDIF()

IF(CONFIG_TFLM_OPTIMIZED_KERNEL_INCLUDE_DIR)
    LIST(APPEND TFLM_EXTRA_KERNEL_INCS ${CONFIG_TFLM_OPTIMIZED_KERNEL_INCLUDE_DIR})
ENDIF()

IF(CONFIG_TFLM_OPTIMIZED_KERNEL_DEPS)
    LIST(APPEND TFLM_EXTRA_KERNEL_DEPS ${CONFIG_TFLM_OPTIMIZED_KERNEL_DEPS})
ENDIF()

SET(CUSTOM_QUANT_SRC ${TFL_SRC}/experimental/custom_quantization_util.cc)
IF(EXISTS ${CUSTOM_QUANT_SRC})
    SET(OPT_SRC ${CUSTOM_QUANT_SRC})
ENDIF()

SET(TFLM_REFERENCE_KERNEL_SRCS
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
)

# This files only exists in newer versions of TF
IF(EXISTS ${TFLM_SRC}/kernels/conv_common.cc)
    LIST(APPEND TFLM_REFERENCE_KERNEL_SRCS ${TFLM_SRC}/kernels/conv_common.cc)
ENDIF()
IF(EXISTS ${TFLM_SRC}/kernels/depthwise_conv_common.cc)
    LIST(APPEND TFLM_REFERENCE_KERNEL_SRCS ${TFLM_SRC}/kernels/depthwise_conv_common.cc)
ENDIF()
IF(EXISTS ${TFLM_SRC}/kernels/fully_connected_common.cc)
    LIST(APPEND TFLM_REFERENCE_KERNEL_SRCS ${TFLM_SRC}/kernels/fully_connected_common.cc)
ENDIF()
IF(EXISTS ${TFLM_SRC}/kernels/quantize_common.cc)
    LIST(APPEND TFLM_REFERENCE_KERNEL_SRCS ${TFLM_SRC}/kernels/quantize_common.cc)
ENDIF()
IF(EXISTS ${TFLM_SRC}/kernels/softmax_common.cc)
    LIST(APPEND TFLM_REFERENCE_KERNEL_SRCS ${TFLM_SRC}/kernels/softmax_common.cc)
ENDIF()
IF(EXISTS ${TFLM_SRC}/kernels/svdf_common.cc)
    LIST(APPEND TFLM_REFERENCE_KERNEL_SRCS ${TFLM_SRC}/kernels/svdf_common.cc)
ENDIF()
IF(EXISTS ${TFLM_SRC}/kernels/pooling_common.cc)
    LIST(APPEND TFLM_REFERENCE_KERNEL_SRCS ${TFLM_SRC}/kernels/pooling_common.cc)
ENDIF()
IF(EXISTS ${TFLM_SRC}/kernels/add_common.cc)
    LIST(APPEND TFLM_REFERENCE_KERNEL_SRCS ${TFLM_SRC}/kernels/add_common.cc)
ENDIF()
IF(EXISTS ${TFLM_SRC}/kernels/dequantize_common.cc)
    LIST(APPEND TFLM_REFERENCE_KERNEL_SRCS ${TFLM_SRC}/kernels/dequantize_common.cc)
ENDIF()
IF(EXISTS ${TFLM_SRC}/kernels/detection_postprocess.cc)
    LIST(APPEND TFLM_REFERENCE_KERNEL_SRCS ${TFLM_SRC}/kernels/detection_postprocess.cc)
ENDIF()

FOREACH(src ${TFLM_EXTRA_KERNEL_SRCS})
    GET_FILENAME_COMPONENT(src_name ${src} NAME)
    IF(${src_name} MATCHES ".*_test.*")
        LIST(REMOVE_ITEM TFLM_EXTRA_KERNEL_SRCS ${src})
    ELSE()
        SET(src_path "${TFLM_SRC}/kernels/${src_name}")
        LIST(FIND TFLM_REFERENCE_KERNEL_SRCS ${src_path} TFLM_KERNEL_SRCS_FOUND_INDEX)
        IF(${TFLM_KERNEL_SRCS_FOUND_INDEX} GREATER_EQUAL 0)
            MESSAGE(STATUS "Replacing TFLM version of ${src_name} by optimized variant...")
            LIST(REMOVE_ITEM TFLM_REFERENCE_KERNEL_SRCS ${src_path})
        ENDIF()
    ENDIF()
ENDFOREACH()

# This files only exists in newer versions of TF
IF(EXISTS ${TFL_SRC}/schema/schema_utils.cc)
    LIST(APPEND OPT_SRC ${TFL_SRC}/schema/schema_utils.cc)
ENDIF()
IF(EXISTS ${TFLM_SRC}/micro_context.cc)
    LIST(APPEND OPT_SRC ${TFLM_SRC}/micro_context.cc)
ENDIF()
IF(EXISTS ${TFLM_SRC}/micro_graph.cc)
    LIST(APPEND OPT_SRC ${TFLM_SRC}/micro_graph.cc)
ENDIF()
IF(EXISTS ${TFLM_SRC}/flatbuffer_utils.cc)
    LIST(APPEND OPT_SRC ${TFLM_SRC}/flatbuffer_utils.cc)
ENDIF()

ADD_LIBRARY(
    tflm STATIC
    # Not really needed?
    ${TFLM_SRC}/micro_error_reporter.cc
    ${TFLM_SRC}/debug_log.cc
    ${TFLM_SRC}/micro_string.cc
    # For reporter->Report
    ${CONFIG_TF_DIR}/tensorflow/lite/core/api/error_reporter.cc
    # Kernels
    ${TFLM_REFERENCE_KERNEL_SRCS}
    ${TFLM_EXTRA_KERNEL_SRCS}
    # Kernel deps
    ${TFLM_SRC}/kernels/kernel_util.cc
    ${TFLM_SRC}/all_ops_resolver.cc
    ${TFLM_SRC}/micro_utils.cc
    ${TFL_SRC}/kernels/internal/quantization_util.cc
    ${TFL_SRC}/kernels/kernel_util.cc
    ${TFLM_SRC}/micro_interpreter.cc
    ${TFLM_SRC}/micro_allocator.cc
    ${TFLM_SRC}/micro_allocation_info.cc
    ${TFLM_SRC}/micro_resource_variable.cc
    ${TFLM_SRC}/arena_allocator/simple_memory_allocator.cc
    ${TFLM_SRC}/memory_helpers.cc
    ${TFLM_SRC}/memory_planner/greedy_memory_planner.cc
    ${TFL_SRC}/core/api/tensor_utils.cc
    ${TFL_SRC}/core/api/flatbuffer_conversions.cc
    ${TFL_SRC}/core/api/op_resolver.cc
    ${TFL_SRC}/c/common.cc
    ${OPT_SRC}
)

IF(TFLM_EXTRA_KERNEL_LIBS)
    TARGET_LINK_LIBRARIES(tflm PUBLIC ${TFLM_EXTRA_KERNEL_LIBS})
ENDIF()
TARGET_LINK_LIBRARIES(tflm PUBLIC m)

IF(TFLM_OPTIMIZED_KERNEL_DEPS)
    ADD_DEPENDENCIES(tflm ${TFLM_OPTIMIZED_KERNEL_DEPS})
ENDIF()

# cmake-format: off
TARGET_INCLUDE_DIRECTORIES(tflm PUBLIC
    ${CONFIG_TF_DIR}
    ${TFLD_SRC}/flatbuffers/include
    ${TFLD_SRC}/gemmlowp
    ${TFLD_SRC}/ruy
    ${TFLM_EXTRA_KERNEL_INCS}
)
# cmake-format: on

# cmake-format: off
TARGET_COMPILE_DEFINITIONS(tflm PUBLIC
    TF_LITE_USE_GLOBAL_CMATH_FUNCTIONS
    TF_LITE_USE_GLOBAL_MAX
    TF_LITE_USE_GLOBAL_MIN
    TF_LITE_STATIC_MEMORY
    TFLITE_EMULATE_FLOAT
    "$<$<CONFIG:RELEASE>:TF_LITE_STRIP_ERROR_STRINGS>"
    ${TFLM_OPTIMIZED_KERNEL_UPPER}
)

# Workaround for the following issue which does not envolve patching the tflite-micro codebase:

# .../micro_error_reporter.cc: In function 'tflite::ErrorReporter* tflite::GetMicroErrorReporter()':
# .../micro_error_reporter.cc:58:76: error: 'static void tflite::MicroErrorReporter::operator delete(void*)' is private within this context
#    58 |     error_reporter_ = new (micro_error_reporter_buffer) MicroErrorReporter();
#       |                                                                            ^
# In file included from .../micro_error_reporter.h:21,
#                  from .../micro_error_reporter.cc:16:
# .../compatibility.h:27:8: note: declared private here
#    27 |   void operator delete(void* p) {}
#       |        ^~~~~~~~
# .../micro_error_reporter.h:51:3: note: in expansion of macro 'TF_LITE_REMOVE_VIRTUAL_DELETE'
#    51 |   TF_LITE_REMOVE_VIRTUAL_DELETE
#
# If disabling the exceptions has major disadvantages needs to be investigated.

TARGET_COMPILE_OPTIONS(tflm PUBLIC
    -fno-exceptions
)
# cmake-format: on
