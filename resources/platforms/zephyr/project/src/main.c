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
  // uint32_t start_time;
  // uint32_t start_time_;
  // uint32_t stop_time;
  // uint32_t stop_time_;
  // uint32_t cycles_spent;
  // uint64_t cycles_spent_;
  // uint32_t nanoseconds_spent;
  // uint64_t nanoseconds_spent_;
  /* capture initial time stamp */
  // start_time = k_cycle_get_32();
  // start_time_ = k_cycle_get_64();
  timing_t start_time, end_time;
  // printk("start_time_=%llu", start_time_);
  start_time = timing_counter_get();
  mlif_run();
  // stop_time = k_cycle_get_32();
  // stop_time_ = k_cycle_get_64();
  end_time = timing_counter_get();
  uint64_t cycles = timing_cycles_get(&start_time, &end_time);
  uint64_t ns_spent = timing_cycles_to_ns(cycles);
  // printk("stop_time_=%llu", stop_time_);

  /* compute how long the work took (assumes no counter rollover) */
  // if (start_time >= stop_time) {
  //     cycles_spent_ = stop_time - start_time;
  // } else {
  //     cycles_spent_ = stop_time + ((1 << 31) - start_time);
  // }
  // cycles_spent_ = stop_time_ - start_time_;
  // nanoseconds_spent = k_cyc_to_ns_ceil32(cycles_spent);
  // nanoseconds_spent_ = k_cyc_to_ns_ceil64(cycles_spent_);
  // printk("Total Time: %u us\n", (uint32_t) (nanoseconds_spent/1000));
  printk("Total Time: %llu us\n", (ns_spent/1000));
  // printk("Total Cycles: %u\n", cycles_spent);
  printk("Total Cycles: %llu\n", cycles);
  printk("MLonMCU: STOP\n");
  // deinit_target();
  sys_reboot(0);
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
