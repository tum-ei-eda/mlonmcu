#include "bench.h"

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
