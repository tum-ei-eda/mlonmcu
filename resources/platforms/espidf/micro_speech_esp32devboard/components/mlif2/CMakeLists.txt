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

SET(MLONMCU_CODEGEN_DIR ${CONFIG_MLONMCU_CODEGEN_DIR})
SET(MLONMCU_FRAMEWORK ${CONFIG_MLONMCU_FRAMEWORK})
SET(MLONMCU_BACKEND ${CONFIG_MLONMCU_BACKEND})


SET(MLIF_SRCS ml_interface.c)
SET(CODEGEN_INCS ${MLONMCU_CODEGEN_DIR})
IF("${MLONMCU_FRAMEWORK}" STREQUAL "tflm")
    SET(LIB_NAME tflite-lib)
    LIST(APPEND MLIF_SRCS ml_interface_tflm.cc)
    FILE(GLOB CODEGEN_SRCS
        ${MLONMCU_CODEGEN_DIR}/*.cc
        ${MLONMCU_CODEGEN_DIR}/*.c
    )

ELSEIF("${MLONMCU_FRAMEWORK}" STREQUAL "tvm")
    SET(LIB_NAME tvm-lib)
    LIST(APPEND MLIF_SRCS ml_interface_tvm.c)
    FILE(GLOB CODEGEN_SRCS
        ${MLONMCU_CODEGEN_DIR}/codegen/host/src/*.c
        ${MLONMCU_CODEGEN_DIR}/codegen/host/src/*.cc
    )
    IF(EXISTS ${MLONMCU_CODEGEN_DIR}/codegen/host/include/)
        LIST(APPEND CODEGEN_INCS ${MLONMCU_CODEGEN_DIR}/codegen/host/include/)
    ENDIF()

    IF ("${MLONMCU_BACKEND}" STREQUAL "tvmaot")
        LIST(APPEND CODEGEN_SRCS ${MLONMCU_CODEGEN_DIR}/aot_wrapper.c)
    ELSEIF ("${MLONMCU_BACKEND}" STREQUAL "tvmrt")
        LIST(APPEND CODEGEN_SRCS ${MLONMCU_CODEGEN_DIR}/rt_wrapper.c)
    ELSEIF ("${MLONMCU_BACKEND}" STREQUAL "tvmcg")
        LIST(APPEND CODEGEN_SRCS ${MLONMCU_CODEGEN_DIR}/staticrt.c)
    ELSE()
        MESSAGE(WARNING "Invalid Backend: ${MLONMCU_BACKEND}")
    ENDIF()
ELSE()
    MESSAGE(WARNING "Invalid Framework: ${MLONMCU_FRAMEWORK}")
    SET(LIB_NAME tflite-lib tvm-lib)
ENDIF()

# SET(MODEL_SUPPORT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/default_model_support)
# SET(MODEL_SUPPORT_SRCS
#     ${MODEL_SUPPORT_DIR}/process_input.c
#     ${MODEL_SUPPORT_DIR}/process_output.c
# )
# SET(DATA_SRC ${CMAKE_CURRENT_SOURCE_DIR}/default_data.c)

# idf_component_register(SRCS ${CODEGEN_SRCS} ${MLIF_SRCS} ${MODEL_SUPPORT_SRCS} ${DATA_SRC}
#                     INCLUDE_DIRS ${CODEGEN_INCS} ${MODEL_SUPPORT_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include
#                     REQUIRES ${LIB_NAME})
# SET(MLIF_EXTRA_INCS
#   ${CMAKE_CURRENT_SOURCE_DIR}/${MLONMCU_FRAMEWORK}
# )

idf_component_register(SRCS ${CODEGEN_SRCS} ${MLIF_SRCS}
                    INCLUDE_DIRS ${CODEGEN_INCS} ${CMAKE_CURRENT_SOURCE_DIR}/include
                    REQUIRES ${LIB_NAME})

# target_compile_definitions(${COMPONENT_LIB} PUBLIC NUM_RUNS=${NUM_RUNS})
