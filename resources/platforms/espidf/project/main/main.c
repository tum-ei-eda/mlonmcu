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
#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "riscv/csr.h"
#include "esp_system.h"
#include "esp_timer.h"

#include "ml_interface.h"

// TODO: move pre and post stuff to mlif or somewhere else?

void app_main(void)
{
    RV_WRITE_CSR(CSR_PCER_MACHINE,8);
    printf("MLonMCU: START\n");
    uint64_t us_before = esp_timer_get_time();
    esp_cpu_ccount_t cc_before = esp_cpu_get_ccount();
    mlif_run();
    uint64_t us_after = esp_timer_get_time();
    uint64_t us_diff = us_after - us_before;
    esp_cpu_ccount_t cc_after = esp_cpu_get_ccount();
    // Warning:
    // 32 bit printf only for newlib nano...
    printf("Total Time: %u us\n", (uint32_t)us_diff);
    printf("Total Cycles: %u\n", (uint32_t)(cc_after - cc_before));  // TODO: overflow possible?
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
