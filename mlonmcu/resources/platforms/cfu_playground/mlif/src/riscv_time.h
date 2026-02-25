#ifndef TARGETLIB_RISCV_TIME_H
#define TARGETLIB_RISCV_TIME_H

/**
 * @brief Returns the number of clock cycles executed by the processor.
 * Overflows after 42.9 seconds on a 100 MIPS and with CPI = 1 (OVPsim).
 */
static inline uint32_t rdcycle(void) {
#if defined(__riscv) || defined(__riscv__)
  uint32_t cycles;
  __asm__ volatile("rdcycle %0" : "=r"(cycles));
  return cycles;
#else
  return 0;
#endif
}

/**
 * @brief Returns the full 64bit register cycle register, which holds the
 * number of clock cycles executed by the processor.
 */
static inline uint64_t rdcycle64()
{
#if defined(__riscv) || defined(__riscv__)
#if __riscv_xlen == 32
    uint32_t cycles;
    uint32_t cyclesh1;
    uint32_t cyclesh2;

    /* Reads are not atomic. So ensure, that we are never reading inconsistent
     * values from the 64bit hardware register. */
    do
    {
        __asm__ volatile("rdcycleh %0" : "=r"(cyclesh1));
        __asm__ volatile("rdcycle %0" : "=r"(cycles));
        __asm__ volatile("rdcycleh %0" : "=r"(cyclesh2));
    } while (cyclesh1 != cyclesh2);

    return (((uint64_t)cyclesh1) << 32) | cycles;
#else
    uint64_t cycles;
    __asm__ volatile("rdcycle %0" : "=r"(cycles));
    return cycles;
#endif
#else
    return 0;
#endif
}

/* How many time cycles (rdtime) per second (OVPsim and Spike). */
#define RDTIME_PER_SECOND 1000000UL

/**
 * @brief Returns wall-clock real time that has passed from an arbitrary start time in the past.
 */
static inline uint32_t rdtime(void) {
#if defined(__riscv) || defined(__riscv__)
  uint32_t time;
  __asm__ volatile("rdtime %0" : "=r"(time));
  return time;
#else
  return 0;
#endif
}

/**
 * @brief Returns the number of instructions retired by the processor.
 */
static inline uint32_t rdinstret(void) {
#if defined(__riscv) || defined(__riscv__)
  uint32_t instret;
  __asm__ volatile("rdinstret %0" : "=r"(instret));
  return instret;
#else
  return 0;
#endif
}

/**
 * @brief Returns the full 64bit register cycle register, which holds the
 * number of instructions retired by the processor.
 */
static inline uint64_t rdinstret64()
{
#if defined(__riscv) || defined(__riscv__)
#if __riscv_xlen == 32
    uint32_t instret;
    uint32_t instreth1;
    uint32_t instreth2;

    /* Reads are not atomic. So ensure, that we are never reading inconsistent
     * values from the 64bit hardware register. */
    do
    {
        __asm__ volatile("rdinstreth %0" : "=r"(instreth1));
        __asm__ volatile("rdinstret %0" : "=r"(instret));
        __asm__ volatile("rdinstreth %0" : "=r"(instreth2));
    } while (instreth1 != instreth2);

    return (((uint64_t)instreth1) << 32) | instret;
#else
    uint64_t instret;
    __asm__ volatile("rdinstret %0" : "=r"(instret));
    return instret;
#endif
#else
    return 0;
#endif
}

#endif  // TARGETLIB_RISCV_TIME_H
