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

SET(TVM_SRC ${CONFIG_TVM_DIR})

IF (NOT ${TVM_SRC} STREQUAL "")

SET(TVM_ALIGNMENT_BYTES 4)

# cmake-format: off
SET(TVM_HEADERS
    ${TVM_SRC}/include
    ${TVM_SRC}/3rdparty/dlpack/include
    ${TVM_SRC}/apps/bundle_deploy
    ${TVM_SRC}/src/runtime/crt/include
)
# cmake-format: on

IF(TVM_CRT_CONFIG_DIR)
    LIST(APPEND TVM_HEADERS ${TVM_CRT_CONFIG_DIR})
ELSE()
    LIST(APPEND TVM_HEADERS ${TVM_SRC}/apps/bundle_deploy/crt_config)
ENDIF()

    idf_component_register(
            SRCS
                ${TVM_SRC}/src/runtime/crt/common/crt_backend_api.c
                ${TVM_SRC}/src/runtime/crt/memory/stack_allocator.c
                ${TVM_SRC}/src/runtime/crt/common/crt_runtime_api.c
                ${TVM_SRC}/src/runtime/crt/common/func_registry.c
                ${TVM_SRC}/src/runtime/crt/memory/page_allocator.c
                ${TVM_SRC}/src/runtime/crt/common/ndarray.c
                ${TVM_SRC}/src/runtime/crt/common/packed_func.c
                ${TVM_SRC}/src/runtime/crt/graph_executor/graph_executor.c
                ${TVM_SRC}/src/runtime/crt/graph_executor/load_json.c
            INCLUDE_DIRS
                ${TVM_HEADERS}
    )


component_compile_options(-Wno-error=unused-but-set-parameter)

TARGET_INCLUDE_DIRECTORIES(${COMPONENT_LIB} PUBLIC
    -DTVM_RUNTIME_ALLOC_ALIGNMENT_BYTES=${TVM_ALIGNMENT_BYTES}
)
ELSE()
    idf_component_register()
ENDIF()

