/*
 * Copyright (c) 2022 TUM Department of Electrical and Computer Engineering.
 *
 * This file is part of the MicroKWS project.
 * See https://gitlab.lrz.de/de-tum-ei-eda-esl/ESD4ML/micro-kws for further
 * info.
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


#include <cstdio>
#include <cstring>
#include <cstdint>
#include "bench.h"
#include "riscv/csr.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "esp_cpu.h"
#include "esp_idf_version.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"


BENCH_DATA_DECL(BENCH_METRIC_CYCLES)
BENCH_DATA_DECL(BENCH_METRIC_INSTRUCTIONS)

#ifdef TIME
BENCH_DATA_DECL(BENCH_METRIC_TIME)
#endif

static char* bench_names[MAX_NUM_BENCH] = {
    BENCH_NAME_0,
    BENCH_NAME_1,
    BENCH_NAME_2,
};

void target_init() {
    printf("MLonMCU: START\n");
    RV_WRITE_CSR(CSR_PCER_MACHINE,PCER_INIT_VAL); 
    long pcer = RV_READ_CSR(CSR_PCER_MACHINE);
    printf("PCER = %ld\n", pcer);
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
    esp_cpu_ccount_t cc = RV_READ_CSR(0x7e2);
#else
    esp_cpu_cycle_count_t cc = RV_READ_CSR(0x7e2);
#endif
    return cc;
}

// uint64_t target_instructions() {
uint32_t target_instructions() {
    // Warning: 32-bit only!
#if ESP_IDF_VERSION < ESP_IDF_VERSION_VAL(5, 0, 0)
    esp_cpu_ccount_t cc = RV_READ_CSR(0x7e2);
#else
    esp_cpu_cycle_count_t cc = RV_READ_CSR(0x7e2);
#endif
    return cc;
}

// uint64_t target_time() {
uint64_t target_time() {
    uint64_t us = esp_timer_get_time();
    // return us / 1000000.0;
    return us;
}

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

    PRINT_BENCH(BENCH_METRIC_CYCLES)
    PRINT_BENCH(BENCH_METRIC_INSTRUCTIONS)

#ifdef TIME
    PRINT_BENCH(BENCH_METRIC_TIME)
#endif
}


