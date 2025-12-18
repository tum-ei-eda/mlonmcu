/*
 * Copyright (c) 2022 TUM Department of Electrical and Computer Engineering.
 *
 * This file is part of MLonMCU.
 * See https://github.com/tum-ei-eda/mlonmcu.git for further info.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "riscv/csr.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "esp_cpu.h"
#include "esp_idf_version.h"

#include "ml_interface.h"

#define mlonmcu_printf printf

#define HAS_CYCLES 1
#define HAS_INSTRUCTIONS 0
#define HAS_TIME 1

// TODO: move pre and post stuff to mlif or somewhere else?

void target_init() {
    printf("MLonMCU: START\n");
    RV_WRITE_CSR(CSR_PCER_MACHINE,8);
}

void target_deinit() {
    printf("MLonMCU: STOP\n");
    // printf("Minimum free heap size: %d bytes\n", esp_get_minimum_free_heap_size());
    fflush(stdout);

    for (int i = 10; i >= 0; i--) {
        printf("Restarting in %d seconds...\n", i);
        vTaskDelay(1000 / portTICK_PERIOD_MS);
    }
    printf("Restarting now.\n");
    fflush(stdout);
    esp_restart();
}

// uint64_t target_cycles() {
uint32_t target_cycles() {
    // Warning: 32-bit only!
#if ESP_IDF_VERSION < ESP_IDF_VERSION_VAL(5, 0, 0)
    esp_cpu_ccount_t cc = esp_cpu_get_ccount();
#else
   esp_cpu_cycle_count_t cc = esp_cpu_get_cycle_count();
#endif
    return cc;
}

// uint64_t target_instructions() {
uint32_t target_instructions() {
    return 0;  // TODO: can only benchmark instructions OR cycles!
}

// uint64_t target_time() {
uint64_t target_time() {
    uint64_t us = esp_timer_get_time();
    // return us / 1000000.0;
    return us;
}

#define MAX_NUM_BENCH 3

#define INIT 0
#define RUN 1
#define TOTAL 2

#if HAS_CYCLES && HAS_INSTRUCTIONS && HAS_TIME

#define MAX_METRICS_IDX 2
#define CYCLES 0
#define INSTRUCTIONS 1
#define TIME 2

#elif HAS_CYCLES && HAS_INSTRUCTIONS

#define CYCLES 0
#define INSTRUCTIONS 1
#define MAX_METRICS_IDX 1

#elif HAS_CYCLES && HAS_TIME

#define CYCLES 0
#define TIME 1
#define MAX_METRICS_IDX 1

#elif HAS_INSTRUCTIONS && HAS_TIME

#define INSTRUCTIONS 0
#define TIME 1
#define MAX_METRICS_IDX 1

#elif HAS_CYCLES

#define CYCLES 0
#define MAX_METRICS_IDX 0

#elif HAS_INSTRUCTIONS

#define INSTRUCTIONS 0
#define MAX_METRICS_IDX 0

#elif HAS_TIME

#define TIME 0
#define MAX_METRICS_IDX 0

#else

#define MAX_METRICS_IDX -1

#endif

#define BENCH_NAME_0 "Setup"
#define BENCH_NAME_1 "Run"
#define BENCH_NAME_2 "Total"

#define BENCH_METRIC_CYCLES 0
#define BENCH_METRIC_INSTRUCTIONS 1
#define BENCH_METRIC_TIME 2
#define BENCH_METRIC_0 "Cycles"
#define BENCH_METRIC_1 "Instructions"
#define BENCH_METRIC_2 "Runtime [us]"

// #define BENCH_TYPE_0 uint64_t
// #define BENCH_TYPE_1 uint64_t
// #define BENCH_TYPE_2 uint64_t
#define BENCH_TYPE_0 uint32_t
#define BENCH_TYPE_1 uint32_t

#ifdef CONFIG_NEWLIB_NANO_FORMAT
#define BENCH_TYPE_2 uint32_t
#else
#define BENCH_TYPE_2 uint64_t
#endif

// #define BENCH_FMT_0 PRIu64
// #define BENCH_FMT_1 PRIu64
// #define BENCH_FMT_2 PRIu64
#define BENCH_FMT_0 PRIu32
#define BENCH_FMT_1 PRIu32
#ifdef CONFIG_NEWLIB_NANO_FORMAT
#define BENCH_FMT_2 PRIu32
#else
#define BENCH_FMT_2 PRIu64
#endif

#define BENCH_FUNC_0 target_cycles
#define BENCH_FUNC_1 target_instructions
#define BENCH_FUNC_2 target_time

#define BENCH_NAME2(index) BENCH_NAME_ ## index
#define BENCH_NAME(index) BENCH_NAME2(index)
#define BENCH_METRIC2(index) BENCH_METRIC_ ## index
#define BENCH_METRIC(index) BENCH_METRIC2(index)
#define BENCH_TYPE2(index) BENCH_TYPE_ ## index
#define BENCH_TYPE(index) BENCH_TYPE2(index)
#define BENCH_FMT2(index) BENCH_FMT_ ## index
#define BENCH_FMT(index) BENCH_FMT2(index)
#define BENCH_FUNC2(index) BENCH_FUNC_ ## index
#define BENCH_FUNC(index) BENCH_FUNC2(index)

// #define BENCH_DATA3(metric) BENCH_METRIC(metric)
#define BENCH_DATA2(metric) temp_ ## metric
#define BENCH_DATA(metric) BENCH_DATA2(metric)
#define BENCH_DATA_DECL(metric) static BENCH_TYPE(metric) BENCH_DATA(metric)[MAX_NUM_BENCH] = {0};

#ifdef CYCLES
BENCH_DATA_DECL(BENCH_METRIC_CYCLES)
#endif
#ifdef INSTRUCTIONS
BENCH_DATA_DECL(BENCH_METRIC_INSTRUCTIONS)
#endif
#ifdef TIME
BENCH_DATA_DECL(BENCH_METRIC_TIME)
#endif

#define PRINT_BENCH(metric) mlonmcu_printf("# %s " BENCH_METRIC(metric) ": %" BENCH_FMT(metric) "\n", bench_names[index], BENCH_DATA(metric)[index]);

static char* bench_names[MAX_NUM_BENCH] = {
    BENCH_NAME_0,
    BENCH_NAME_1,
    BENCH_NAME_2,
};

void start_bench(size_t index) {
#ifdef CYCLES
    BENCH_TYPE(BENCH_METRIC_CYCLES) cycles = BENCH_FUNC(BENCH_METRIC_CYCLES)();
#endif
#ifdef INSTRUCTIONS
    BENCH_TYPE(BENCH_METRIC_INSTRUCTIONS) instructions = BENCH_FUNC(BENCH_METRIC_INSTRUCTIONS)();
#endif
#ifdef TIME
    BENCH_TYPE(BENCH_METRIC_TIME) time = BENCH_FUNC(BENCH_METRIC_TIME)();
#endif
#ifdef CYCLES
    BENCH_DATA(BENCH_METRIC_CYCLES)[index] = cycles;
#endif
#ifdef INSTRUCTIONS
    BENCH_DATA(BENCH_METRIC_INSTRUCTIONS)[index] = instructions;
#endif
#ifdef TIME
    BENCH_DATA(BENCH_METRIC_TIME)[index] = time;
#endif
}

void stop_bench(size_t index) {
#ifdef CYCLES
    BENCH_TYPE(BENCH_METRIC_CYCLES) cycles = BENCH_FUNC(BENCH_METRIC_CYCLES)();
#endif
#ifdef INSTRUCTIONS
    BENCH_TYPE(BENCH_METRIC_INSTRUCTIONS) instructions = BENCH_FUNC(BENCH_METRIC_INSTRUCTIONS)();
#endif
#ifdef TIME
    BENCH_TYPE(BENCH_METRIC_TIME) time = BENCH_FUNC(BENCH_METRIC_TIME)();
#endif
    // TODO: check for overflow
#ifdef CYCLES
    BENCH_DATA(BENCH_METRIC_CYCLES)[index] = cycles - BENCH_DATA(BENCH_METRIC_CYCLES)[index];
#endif
#ifdef INSTRUCTIONS
    BENCH_DATA(BENCH_METRIC_INSTRUCTIONS)[index] = instructions - BENCH_DATA(BENCH_METRIC_INSTRUCTIONS)[index];
#endif
#ifdef TIME
    BENCH_DATA(BENCH_METRIC_TIME)[index] = time - BENCH_DATA(BENCH_METRIC_TIME)[index];
#endif
}

void print_bench(size_t index) {
#ifdef CYCLES
    PRINT_BENCH(BENCH_METRIC_CYCLES)
#endif
#ifdef INSTRUCTIONS
    PRINT_BENCH(BENCH_METRIC_INSTRUCTIONS)
#endif
#ifdef TIME
    PRINT_BENCH(BENCH_METRIC_TIME)
#endif
}

#ifdef __cplusplus
extern "C" {
#endif
int mlonmcu_init();
int mlonmcu_run();
int mlonmcu_check();
int mlonmcu_deinit();
#ifdef __cplusplus
}
#endif

#define EXIT_MLIF_BASE (0x10)
#define EXIT_MLIF_INVALID_SIZE (EXIT_MLIF_BASE + 1)
#define EXIT_MLIF_MISSMATCH (EXIT_MLIF_BASE + 2)

void mlonmcu_exit(int status) {
    mlonmcu_printf("MLONMCU EXIT: %d\n", status);
    // exit(status);
}

//     uint64_t us_before = esp_timer_get_time();
//     // printf("VER: %s, %x, %d, %d, %d\n", IDF_VER, ESP_IDF_VERSION, ESP_IDF_VERSION, ESP_IDF_VERSION_MINOR, ESP_IDF_VERSION_PATCH);
//  #if ESP_IDF_VERSION < ESP_IDF_VERSION_VAL(5, 0, 0)
//      esp_cpu_ccount_t cc_before = esp_cpu_get_ccount();
//  #else
//     esp_cpu_cycle_count_t cc_before = esp_cpu_get_cycle_count();
//  #endif
//     mlif_run();
//     uint64_t us_after = esp_timer_get_time();
//     uint64_t us_diff = us_after - us_before;
//  #if ESP_IDF_VERSION < ESP_IDF_VERSION_VAL(5, 0, 0)
//     esp_cpu_ccount_t cc_after = esp_cpu_get_ccount();
//  #else
//     esp_cpu_cycle_count_t cc_after = esp_cpu_get_cycle_count();
//  #endif
//     // Warning:
//     // 32 bit printf only for newlib nano...
//     printf("Total Time: %" PRIu32 " us\n", (uint32_t)us_diff);
//     printf("Total Cycles: %" PRIu32 "\n", (uint32_t)(cc_after - cc_before));  // TODO: overflow possible?

void app_main(void)
{
    int ret = 0;
    target_init();
    start_bench(TOTAL);
    start_bench(INIT);
    ret = mlonmcu_init();
    stop_bench(INIT);
    if (ret) {
      goto cleanup;
    }
    start_bench(RUN);
    ret = mlonmcu_run();
    stop_bench(RUN);
    if (ret) {
      goto cleanup;
    }
#ifndef MLONMCU_SKIP_CHECK
    ret = mlonmcu_check();
    if (ret) {
      goto cleanup;
    }
#endif  // !MLONMCU_SKIP_CHECK
    // start_bench(DEINIT);
    ret = mlonmcu_deinit();
    // stop_bench(DEINIT);
    if (ret) {
      goto cleanup;
    }

cleanup:
    stop_bench(TOTAL);

    // post
    print_bench(INIT);
    print_bench(RUN);
    // print_bench(DEINIT);
    print_bench(TOTAL);
    mlonmcu_printf("Program finish.\n");
    mlonmcu_exit(ret);
    target_deinit();
}
