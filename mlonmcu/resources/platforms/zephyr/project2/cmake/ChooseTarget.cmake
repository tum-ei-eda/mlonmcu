# New target architectures and systems should be added here Make sure to define the CMAKE_TOOLCHAIN_FILE for
# cross-compilation

# Default implementation of the macro points to the original ADD_LIBRARY function
MACRO(COMMON_ADD_LIBRARY TARGET_NAME)
    ADD_LIBRARY(${TARGET_NAME} ${ARGN})
ENDMACRO()

MACRO(COMMON_ADD_EXECUTABLE TARGET_NAME)
    ADD_EXECUTABLE(${TARGET_NAME} ${ARGN})
ENDMACRO()

SET(TOOLCHAIN_FILE "")
IF(TARGET_SYSTEM)
    INCLUDE(targets/${TARGET_SYSTEM})
    IF(TOOLCHAIN)
        SET(TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/toolchains/${TOOLCHAIN}_${TARGET_SYSTEM}.cmake")
        INCLUDE(${TOOLCHAIN_FILE})
    ENDIF()
ENDIF()

# set(CMAKE_EXECUTABLE_SUFFIX_C .elf)
