#ifndef LIB_BENCH_BENCH_H
#define LIB_BENCH_BENCH_H

#include <stdint.h>
#include <stddef.h>
#include <inttypes.h>
#include <printing.h>

#include "target.h"

#define MAX_NUM_BENCH 4

#define INIT 0
#define RUN 1
#define TOTAL 2
#define DEINIT 3

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
#define BENCH_NAME_3 "Deinit"

#define BENCH_METRIC_CYCLES 0
#define BENCH_METRIC_INSTRUCTIONS 1
#define BENCH_METRIC_TIME 2
#define BENCH_METRIC_0 "Cycles"
#define BENCH_METRIC_1 "Instructions"
#define BENCH_METRIC_2 "Runtime [us]"

#define BENCH_TYPE_0 uint64_t
#define BENCH_TYPE_1 uint64_t
#define BENCH_TYPE_2 uint64_t

#if defined(MLONMCU_TARGET_VICUNA)
#define BENCH_FMT_0 "u"
#define BENCH_FMT_1 "u"
#define BENCH_FMT_2 "u"
#else
#define BENCH_FMT_0 PRIu64
#define BENCH_FMT_1 PRIu64
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
    BENCH_NAME_3,
};


void start_bench(size_t index);
void stop_bench(size_t index);
void print_bench(size_t index);

#endif  // LIB_BENCH_BENCH_H
