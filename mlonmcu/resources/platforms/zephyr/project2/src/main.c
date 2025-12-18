/*
 * Copyright (c) 2012-2014 Wind River Systems, Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <zephyr.h>
#include <stdint.h>
#include <sys/printk.h>
#include <sys/reboot.h>
#include <timing/timing.h>
// #include <zephyr/kernel.h>

// void init_target();
// void deinit_target();

int main() {
  // init_target();
  timing_init();
  timing_start();
  printk("MLonMCU: START\n");
  timing_t start_time, end_time;
  start_time = timing_counter_get();
  mlif_run();
  end_time = timing_counter_get();
  uint64_t cycles = timing_cycles_get(&start_time, &end_time);
  uint64_t ns_spent = timing_cycles_to_ns(cycles);

  printk("Total Time: %llu us\n", (ns_spent/1000));
  printk("Total Cycles: %llu\n", cycles);
  printk("MLonMCU: STOP\n");
  // deinit_target();
  sys_reboot(0);
  return 0;
}
