#include <stdio.h>

// DWT (Data Watchpoint and Trace) registers, only exists on ARM Cortex with a
// DWT unit.
#define KIN1_DWT_CONTROL (*((volatile uint32_t*)0xE0001000))

// DWT Control register.
#define KIN1_DWT_CYCCNTENA_BIT (1UL << 0)

// CYCCNTENA bit in DWT_CONTROL register.
#define KIN1_DWT_CYCCNT (*((volatile uint32_t*)0xE0001004))

// DWT Cycle Counter register.
#define KIN1_DEMCR (*((volatile uint32_t*)0xE000EDFC))

// DEMCR: Debug Exception and Monitor Control Register.
#define KIN1_TRCENA_BIT (1UL << 24)

// Trace enable bit in DEMCR register.
#define KIN1_LAR (*((volatile uint32_t*)0xE0001FB0))

// Unlock access to DWT (ITM, etc.)registers.
#define KIN1_UnlockAccessToDWT() KIN1_LAR = 0xC5ACCE55;

// TRCENA: Enable trace and debug block DEMCR (Debug Exception and Monitor
// Control Register.
#define KIN1_InitCycleCounter() KIN1_DEMCR |= KIN1_TRCENA_BIT

#define KIN1_ResetCycleCounter() KIN1_DWT_CYCCNT = 0
#define KIN1_EnableCycleCounter() KIN1_DWT_CONTROL |= KIN1_DWT_CYCCNTENA_BIT
#define KIN1_DisableCycleCounter() KIN1_DWT_CONTROL &= ~KIN1_DWT_CYCCNTENA_BIT
#define KIN1_GetCycleCounter() KIN1_DWT_CYCCNT

int32_t ticks_per_second() { return 25e6; }

int32_t GetCurrentTimeTicks() { return KIN1_GetCycleCounter(); }

extern void uart_init(void);

static int32_t start_cycles = 0;

void init_target() {
  uart_init();
  KIN1_UnlockAccessToDWT();
  KIN1_InitCycleCounter();
  KIN1_ResetCycleCounter();
  KIN1_EnableCycleCounter();
  int32_t ticks = GetCurrentTimeTicks();
  start_cycles = ticks;
  printf("GetCurrentTimeTicks=%ld\n", ticks);
}

void deinit_target() {
  int32_t stop_cycles = GetCurrentTimeTicks();
  int32_t diff_cycles = stop_cycles - start_cycles;
  // int32_t diff_ms = diff_cycles / (ticks_per_second() / 1000);
  // printf("Total Time: %ld ms\n", diff_ms);
  printf("Total Cycles: %ld\n", diff_cycles);
  printf("EXITTHESIM\n");
  while (1)
    ;
}
