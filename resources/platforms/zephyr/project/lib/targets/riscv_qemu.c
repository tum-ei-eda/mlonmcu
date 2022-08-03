#include <stdio.h>

/* How many cycles (rdcycle) per second (OVPsim and Spike). */
#define RDCYCLE_PER_SECOND 100000000UL

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
static inline uint64_t rdcycle64() {
#if defined(__riscv) || defined(__riscv__)
  uint32_t cycles;
  uint32_t cyclesh1;
  uint32_t cyclesh2;

  /* Reads are not atomic. So ensure, that we are never reading inconsistent
   * values from the 64bit hardware register. */
  do {
    __asm__ volatile("rdcycleh %0" : "=r"(cyclesh1));
    __asm__ volatile("rdcycle %0" : "=r"(cycles));
    __asm__ volatile("rdcycleh %0" : "=r"(cyclesh2));
  } while (cyclesh1 != cyclesh2);

  return (((uint64_t)cyclesh1) << 32) | cycles;
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

static uint64_t start_cycles = 0;

void init_target() {
  enable_fext();
#ifdef USE_VEXT
  enable_vext();
#endif
  start_cycles = rdcycle64();
}

void deinit_target() {
  uint64_t stop_cycles = rdcycle64();
  uint64_t diff_cycles = stop_cycles - start_cycles;
  float diff_ms = 0;  // unimplemented (see RDCYCLE_PER_SECOND)
  printf("Total Cycles: %lld\n", stop_cycles - start_cycles);
}
