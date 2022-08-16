SET(TVM_ALIGNMENT_BYTES 4)

# cmake-format: off
SET(TVM_HEADERS
    ${CONFIG_TVM_DIR}/include
    ${CONFIG_TVM_DIR}/3rdparty/dlpack/include
    ${CONFIG_TVM_DIR}/apps/bundle_deploy
    ${CONFIG_TVM_DIR}/src/runtime/crt/include
)
# cmake-format: on

IF(CONFIG_TVM_CRT_CONFIG_DIR)
    LIST(APPEND TVM_HEADERS ${CONFIG_TVM_CRT_CONFIG_DIR})
ELSE()
    LIST(APPEND TVM_HEADERS ${CONFIG_TVM_DIR}/apps/bundle_deploy/crt_config)
ENDIF()

# cmake-format: off
ADD_LIBRARY(tvm_static_rt STATIC
    ${CONFIG_TVM_DIR}/src/runtime/crt/common/crt_runtime_api.c
    ${CONFIG_TVM_DIR}/src/runtime/crt/memory/stack_allocator.c
)
# cmake-format: on

TARGET_INCLUDE_DIRECTORIES(tvm_static_rt PUBLIC ${TVM_HEADERS})

# cmake-format: off
TARGET_COMPILE_DEFINITIONS(tvm_static_rt PUBLIC
    -DTVM_RUNTIME_ALLOC_ALIGNMENT_BYTES=${TVM_ALIGNMENT_BYTES}
)
# cmake-format: on

# cmake-format: off
ADD_LIBRARY(tvm_graph_rt STATIC
    ${CONFIG_TVM_DIR}/src/runtime/crt/common/crt_backend_api.c
    ${CONFIG_TVM_DIR}/src/runtime/crt/common/crt_runtime_api.c
    ${CONFIG_TVM_DIR}/src/runtime/crt/common/func_registry.c
    ${CONFIG_TVM_DIR}/src/runtime/crt/memory/page_allocator.c
    ${CONFIG_TVM_DIR}/src/runtime/crt/memory/stack_allocator.c
    ${CONFIG_TVM_DIR}/src/runtime/crt/common/ndarray.c
    ${CONFIG_TVM_DIR}/src/runtime/crt/common/packed_func.c
    ${CONFIG_TVM_DIR}/src/runtime/crt/graph_executor/graph_executor.c
    ${CONFIG_TVM_DIR}/src/runtime/crt/graph_executor/load_json.c
)
# cmake-format: on

TARGET_INCLUDE_DIRECTORIES(tvm_graph_rt PUBLIC ${TVM_HEADERS})

# cmake-format: off
TARGET_COMPILE_DEFINITIONS(tvm_graph_rt PUBLIC
    -DTVM_RUNTIME_ALLOC_ALIGNMENT_BYTES=${TVM_ALIGNMENT_BYTES}
)
# cmake-format: on

# cmake-format: off
ADD_LIBRARY(tvm_aot_rt STATIC
    ${CONFIG_TVM_DIR}/src/runtime/crt/common/crt_backend_api.c
    ${CONFIG_TVM_DIR}/src/runtime/crt/memory/stack_allocator.c
)
# cmake-format: on

TARGET_INCLUDE_DIRECTORIES(tvm_aot_rt PUBLIC ${TVM_HEADERS})

# cmake-format: off
TARGET_COMPILE_DEFINITIONS(tvm_aot_rt PUBLIC
    -DTVM_RUNTIME_ALLOC_ALIGNMENT_BYTES=${TVM_ALIGNMENT_BYTES}
)
# cmake-format: on
