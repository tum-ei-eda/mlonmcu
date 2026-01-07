PROJECT(generic_mlonmcu LANGUAGES C CXX ASM)

COMMON_ADD_EXECUTABLE(${PROJECT_NAME} main.c)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PUBLIC mlonmcu_support target_support bench support)
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/lib/bench)
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/lib/targets)

SET(MLONMCU_ASMDUMP ON)
SET(MLONMCU_SRCDUMP ON)

FILE(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/dumps)

IF(MLONMCU_ASMDUMP)
    add_custom_command(TARGET generic_mlonmcu POST_BUILD
        COMMAND ${CMAKE_OBJDUMP} -d ${OBJDUMP_EXTRA_ARGS} ${CMAKE_BINARY_DIR}/bin/generic_mlonmcu > ${CMAKE_BINARY_DIR}/dumps/generic_mlonmcu.dump
        COMMENT "Invoking: Objdump (ASM)")
ENDIF()
IF(MLONMCU_SRCDUMP)
    add_custom_command(TARGET generic_mlonmcu POST_BUILD
        COMMAND ${CMAKE_OBJDUMP} -S ${OBJDUMP_EXTRA_ARGS} ${CMAKE_BINARY_DIR}/bin/generic_mlonmcu > ${CMAKE_BINARY_DIR}/dumps/generic_mlonmcu.srcdump
        COMMENT "Invoking: Objdump (SRC)")
ENDIF()
