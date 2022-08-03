/*
 * Copyright (c) 2012-2014 Wind River Systems, Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <zephyr.h>
#include <stdint.h>
#include <sys/printk.h>
#include <sys/reboot.h>
// #include <zephyr/kernel.h>

// void init_target();
// void deinit_target();

int main() {
  // init_target();
  printk("MLonMCU: START\n");
  uint32_t start_time;
  uint32_t stop_time;
  uint32_t cycles_spent;
  uint32_t nanoseconds_spent;
  /* capture initial time stamp */
  start_time = k_cycle_get_32();
  mlif_run();
  stop_time = k_cycle_get_32();

  /* compute how long the work took (assumes no counter rollover) */
  cycles_spent = stop_time - start_time;
  nanoseconds_spent = k_cyc_to_ns_ceil32(cycles_spent);
  printk("Total Time: %u us\n", (uint32_t) (nanoseconds_spent/1000));
  printk("Total Cycles: %u\n", cycles_spent);
  printk("MLonMCU: STOP\n");
  // deinit_target();
  sys_reboot();
  return 0;
}
//
// /* do work for some (short) period of time */
// ...
//
// /* capture final time stamp */
// stop_time = k_cycle_get_32();
//
// /* compute how long the work took (assumes no counter rollover) */
// cycles_spent = stop_time - start_time;
// nanoseconds_spent = SYS_CLOCK_HW_CYCLES_TO_NS(cycles_spent);
