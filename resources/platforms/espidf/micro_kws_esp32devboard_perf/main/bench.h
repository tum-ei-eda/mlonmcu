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

#ifndef BENCH_H
#define BENCH_H

#include <cinttypes>
#include "sdkconfig.h"

#ifndef PCER_INIT_VAL
// See: https://www.espressif.com/sites/default/files/documentation/esp32-c3_technical_reference_manual_en.pdf
// Count cycles by default
#define PCER_INIT_VAL 1
#endif

#define mlonmcu_printf printf

#if PCER_INIT_VAL == 1
#define HAS_CYCLES 1
#define HAS_INSTRUCTIONS 0
#elif PCER_INIT_VAL == 2
#define HAS_CYCLES 0
#define HAS_INSTRUCTIONS 1
#else
#define HAS_CYCLES 1
#define HAS_INSTRUCTIONS 0
#endif

#define HAS_TIME 1

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

#define PRINT_BENCH(metric) mlonmcu_printf("# %s " BENCH_METRIC(metric) ": %" BENCH_FMT(metric) "\n", bench_names[index], BENCH_DATA(metric)[index]);
#define PRINT_BENCH_DUMMY(metric) mlonmcu_printf("# %s " BENCH_METRIC(metric) ": NA\n", bench_names[index]);

void target_init();
void target_deinit();
void start_bench(size_t index);
void stop_bench(size_t index);
void print_bench(size_t index);

#endif // BENCH_H
