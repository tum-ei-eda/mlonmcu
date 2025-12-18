IF(NOT RISCV_VEXT)
    MESSAGE(FATAL_ERROR "RISCV_AUTO_VECTORIZE requires RISCV_VEXT")
ENDIF()
IF(RISCV_RVV_VLEN)
    SET(VLEN ${RISCV_RVV_VLEN})
ELSE()
    SET(VLEN "?")
ENDIF()
 SET(CMAKE_CXX_FLAGS_RELEASE
     "${CMAKE_CXX_FLAGS_RELEASE} \
     -mllvm \
     --riscv-v-vector-bits-min=${VLEN} \
 "
 )
 SET(CMAKE_C_FLAGS_RELEASE
     "${CMAKE_C_FLAGS_RELEASE} \
     -mllvm \
     --riscv-v-vector-bits-min=${VLEN} \
 "
 )
