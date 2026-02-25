#ifndef TARGETLIB_RISCV_UTILS_H
#define TARGETLIB_RISCV_UTILS_H

/**
 * @brief Enables the vector extension (as well as the floating point extension).
 */
static inline void enable_fext(void)
{
#if defined(__riscv) || defined(__riscv__)
    __asm__ volatile("li t0, 1<<13 \n"
                     "csrs mstatus, t0 \n" ::
                         : "t0");
#endif
}

/**
 * @brief Enables the vector extension (as well as the floating point extension).
 */
static inline void enable_vext(void)
{
#if defined(__riscv) || defined(__riscv__)
    __asm__ volatile("li t0, 1<<9+1<<13 \n"
                     "csrs mstatus, t0 \n" ::
                         : "t0");
#endif
}

#endif  // TARGETLIB_RISCV_UTILS_H
