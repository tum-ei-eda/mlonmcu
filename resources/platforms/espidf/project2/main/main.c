/* Hello World Example

   This example code is in the Public Domain (or CC0 licensed, at your option.)

   Unless required by applicable law or agreed to in writing, this
   software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
   CONDITIONS OF ANY KIND, either express or implied.
*/
#include <stdio.h>
#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "esp_private/esp_clk.h"
#include "esp_spi_flash.h"

// TODO: move pre and post stuff to mlif or somewhere else?

void app_main(void)
{
    printf("MLonMCU: START\n");
    uint64_t us_before = esp_timer_get_time();

    /* Print chip information */
    esp_chip_info_t chip_info;
    esp_chip_info(&chip_info);
    printf("This is %s chip with %d CPU core(s), WiFi%s%s, ",
            CONFIG_IDF_TARGET,
            chip_info.cores,
            (chip_info.features & CHIP_FEATURE_BT) ? "/BT" : "",
            (chip_info.features & CHIP_FEATURE_BLE) ? "/BLE" : "");

    printf("silicon revision %d, ", chip_info.revision);

    printf("%dMB %s flash\n", spi_flash_get_chip_size() / (1024 * 1024),
            (chip_info.features & CHIP_FEATURE_EMB_FLASH) ? "embedded" : "external");

    printf("Minimum free heap size: %d bytes\n", esp_get_minimum_free_heap_size());

    uint64_t us_after = esp_timer_get_time();
    uint64_t us_diff = us_after - us_before;
    int cpu_freq = esp_clk_cpu_freq();
    uint64_t cycles_diff = (cpu_freq / 1000000) * us_diff;
    printf("CPU Frequency: %d\n", cpu_freq);
    printf("Total Time: %llu us\n", us_diff);
    printf("Total Cycles: %llu\n", cycles_diff);
    printf("MLonMCU: STOP\n");
    fflush(stdout);

    for (int i = 10; i >= 0; i--) {
        printf("Restarting in %d seconds...\n", i);
        vTaskDelay(1000 / portTICK_PERIOD_MS);
    }
    printf("Restarting now.\n");
    fflush(stdout);
    esp_restart();
}
