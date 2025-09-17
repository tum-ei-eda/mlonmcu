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
    -ftree-vectorize \
    -mriscv-vector-bits=${VLEN} \
"
)
SET(CMAKE_C_FLAGS_RELEASE
    "${CMAKE_C_FLAGS_RELEASE} \
    -ftree-vectorize \
    -mriscv-vector-bits=${VLEN} \
"
)
IF(DEFINED RISCV_AUTO_VECTORIZE_LOOP)
    IF(NOT RISCV_AUTO_VECTORIZE_LOOP)
        SET(CMAKE_CXX_FLAGS_RELEASE
            "${CMAKE_CXX_FLAGS_RELEASE} \
            -fno-tree-loop-vectorize \
        "
        )
        SET(CMAKE_C_FLAGS_RELEASE
            "${CMAKE_C_FLAGS_RELEASE} \
            -fno-tree-loop-vectorize \
        "
        )
    ENDIF()
ENDIF()
IF(DEFINED RISCV_AUTO_VECTORIZE_SLP)
    IF(NOT RISCV_AUTO_VECTORIZE_SLP)
        SET(CMAKE_CXX_FLAGS_RELEASE
            "${CMAKE_CXX_FLAGS_RELEASE} \
            -fno-tree-slp-vectorize \
        "
        )
        SET(CMAKE_C_FLAGS_RELEASE
            "${CMAKE_C_FLAGS_RELEASE} \
            -fno-tree-slp-vectorize \
        "
        )
    ENDIF()
ENDIF()
# Also interesting:
# -mriscv-vector-lmul=<lmul>  Set the vf using lmul in auto-vectorization
# -fsimd-cost-model=[unlimited|dynamic|cheap|very-cheap] Specifies the vectorization cost model for code marked with a simd directive
IF(RISCV_AUTO_VECTORIZE_VERBOSE)
    SET(CMAKE_CXX_FLAGS_RELEASE
        "${CMAKE_CXX_FLAGS_RELEASE} \
        -fopt-info-vec \
        -fopt-info-vec-missed \
    "
    )
    SET(CMAKE_C_FLAGS_RELEASE
        "${CMAKE_C_FLAGS_RELEASE} \
        -fopt-info-vec \
        -fopt-info-vec-missed \
    "
    )
ENDIF()
