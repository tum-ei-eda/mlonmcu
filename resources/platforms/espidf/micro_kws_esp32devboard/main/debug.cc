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

#include "debug.h"

#include <ctype.h>
#include <inttypes.h>

#include <cstring>

#include "audio.h"
#include "driver/uart.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/task.h"
#include "gpio.h"
#include "model_settings.h"

// TODO(fabianpedd): If we had two cores, like on the ESP32, we could run the
// DebugWorker() task on a different core than the main task that runs the
// inference. This would allow debugging with minimal runtime overhead, as only
// the memcpy operation would be added to the main task.

// TODO(fabianpedd): Make the UART_PORT and UART_TX_PIN define, as well as the
// sdkconfig options CONFIG_ESP_CONSOLE_UART_DEFAULT=y and
// CONFIG_ESP_CONSOLE_NONE=y configurable via the menuconfig.
// We need this in order to be able to either transmit the ESP_LOGs and printfs,
// or the debug data to the python script via the onboard UART.

// Don't forget to change the UART_TX_PIN and the CONFIG_ESP_CONSOLE_XXX setting
#define UART_PORT UART_NUM_0
#define UART_BAUDRATE 200000
#define UART_TX_PIN 21                  // default is pin 21 (or pin 7 for external)
#define UART_RX_PIN UART_PIN_NO_CHANGE  // 20

#ifdef CONFIG_MICRO_KWS_MODE_DEBUG
static RingbufHandle_t buf_handle = NULL;
#endif  // CONFIG_MICRO_KWS_MODE_DEBUG

static TaskHandle_t DebugWorkerHandle = NULL;

typedef struct __attribute__((packed)) {
#ifndef CONFIG_MICRO_KWS_MODE_DEBUG_AUDIO
  int8_t feature_data[feature_element_count];
  uint8_t category_data[category_count];
  uint8_t top_category_index;
#else   // CONFIG_MICRO_KWS_MODE_DEBUG_AUDIO
  uint8_t audio_data[AUDIO_PACKET_SIZE];
#endif  // CONFIG_MICRO_KWS_MODE_DEBUG_AUDIO
} debug_data_t;

#ifdef CONFIG_MICRO_KWS_MODE_DEBUG
static void DebugWorker(void* arg) {
  while (true) {
    size_t item_size = 0;
    debug_data_t* data =
        (debug_data_t*)xRingbufferReceive(buf_handle, &item_size, pdMS_TO_TICKS(500));

    if (data == NULL) {
      ESP_LOGE(__FILE__, "ERROR: In xRingbufferReceive() in DebugWorker().");
      return;
    }

    if (item_size != sizeof(debug_data_t)) {
      ESP_LOGE(__FILE__,
               "ERROR: Received size %d from xRingbufferReceive() in "
               "DebugWorker() does not match expected size %d.",
               item_size, sizeof(debug_data_t));
      return;
    }
    // else {
    //   ESP_LOGD(__FILE__, "Received %d bytes from ringbuffer.", item_size);
    // }

    // TODO(fabianpedd): Maybe include the size of the packet or smth similar
    // (maybe even a checksum).

    // Last eight bytes will be 0x00, 0x01, ..., 0x07 in order to help
    // synchronize the UART packet once sent to the host PC.
    uint8_t packet_footer[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07};

    // Total size of packet consists of data and packet footer size
    constexpr size_t total_packet_size = sizeof(debug_data_t) + sizeof(packet_footer);
    int8_t packet_buffer[total_packet_size] = {0};

    // Copy the data from the UART into the packet buffer and append the packet
    // footer to the end.
    memcpy(packet_buffer, data, sizeof(debug_data_t));
    memcpy(packet_buffer + sizeof(debug_data_t), packet_footer, sizeof(packet_footer));

    // Now we can free the data from the internal UART buffer.
    vRingbufferReturnItem(buf_handle, (void*)data);

    // Send debug data via the auxiliary UART to the Python script.
    size_t data_sent = uart_write_bytes(UART_PORT, (int8_t*)packet_buffer, total_packet_size);

    if (data_sent < total_packet_size) {
      ESP_LOGE(__FILE__,
               "ERROR: Only sent %d of %d bytes via uart_write_bytes() in "
               "DebugWorker().",
               data_sent, total_packet_size);
      return;
    }
    // else {
    //   ESP_LOGD(__FILE__, "Sent %d bytes via UART.", data_sent);
    // }
  }
}
#endif  // CONFIG_MICRO_KWS_MODE_DEBUG

#ifdef CONFIG_MICRO_KWS_PRINT_STATS
static void DebugPrintStats(void* arg) {
  (void)arg;
  while (true) {
    char buffer[1024] = {0};
    // TODO(fabianpedd): The vTaskGetRunTimeStats() appears to be broken, in
    // that it only prints up to a certain number of characters. If you have
    // more tasks active they will be cut off/not be displayed.
    vTaskGetRunTimeStats(buffer);
    printf("%s\n", buffer);
    vTaskDelay(CONFIG_MICRO_KWS_PRINT_STATS_INTERVAL / portTICK_PERIOD_MS);
  }
}
#endif  // CONFIG_MICRO_KWS_PRINT_STATS

esp_err_t InitializeDebug() {
#ifdef CONFIG_MICRO_KWS_MODE_DEBUG
  uart_config_t uart_config = {
      .baud_rate = (int)UART_BAUDRATE,
      .data_bits = UART_DATA_8_BITS,
      .parity = UART_PARITY_DISABLE,
      .stop_bits = UART_STOP_BITS_1,
      .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
      .source_clk = UART_SCLK_APB,
  };

  if (uart_driver_install(UART_PORT, 1024, 0, 0, NULL, 0) != ESP_OK) {
    ESP_LOGE(__FILE__, "ERROR: In uart_driver_install() in DebugInit().");
    return ESP_FAIL;
  }

  if (uart_param_config(UART_PORT, &uart_config)) {
    ESP_LOGE(__FILE__, "ERROR: In uart_param_config() in DebugInit().");
    return ESP_FAIL;
  }

  if (uart_set_pin(UART_PORT, UART_TX_PIN, UART_RX_PIN, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE)) {
    ESP_LOGE(__FILE__, "ERROR: In uart_set_pin() in DebugInit().");
    return ESP_FAIL;
  }

#ifndef CONFIG_MICRO_KWS_MODE_DEBUG_AUDIO
  buf_handle = xRingbufferCreate(1024 * 16, RINGBUF_TYPE_NOSPLIT);
#else   // CONFIG_MICRO_KWS_MODE_DEBUG_AUDIO
  buf_handle = xRingbufferCreate(1024 * 32, RINGBUF_TYPE_NOSPLIT);
#endif  // CONFIG_MICRO_KWS_MODE_DEBUG_AUDIO
  if (buf_handle == NULL) {
    ESP_LOGE(__FILE__, "ERROR: In xRingbufferCreate() in DebugInit().");
    return ESP_FAIL;
  }

#ifndef CONFIG_MICRO_KWS_MODE_DEBUG_AUDIO
  if (xTaskCreate(DebugWorker, "DebugWorker", 1024 * 16, NULL, 10, &DebugWorkerHandle) !=
#else   // CONFIG_MICRO_KWS_MODE_DEBUG_AUDIO
  if (xTaskCreate(DebugWorker, "DebugWorker", 1024 * 32, NULL, 10, &DebugWorkerHandle) !=
#endif  // CONFIG_MICRO_KWS_MODE_DEBUG_AUDIO
      pdPASS) {
    ESP_LOGE(__FILE__, "ERROR: In xTaskCreate(DebugWorker) in DebugInit().");
    return ESP_FAIL;
  }
#endif  // CONFIG_MICRO_KWS_MODE_DEBUG

#ifdef CONFIG_MICRO_KWS_PRINT_STATS
  if (xTaskCreate(DebugPrintStats, "DebugPrintStats", 1024 * 4, NULL, 10, NULL) != pdPASS) {
    ESP_LOGE(__FILE__, "ERROR: In xTaskCreate(DebugPrintStats) in DebugInit().");
    return ESP_FAIL;
  }
#endif  // CONFIG_MICRO_KWS_MODE_DEBUG

  return ESP_OK;
}

esp_err_t StopDebug() {
  // TODO(fabianpedd): Also free Ringbuffer and deinstall UART
  vTaskDelete(DebugWorkerHandle);
  return ESP_OK;
}

#ifndef CONFIG_MICRO_KWS_MODE_DEBUG_AUDIO
esp_err_t DebugRun(int8_t* feature_data, uint8_t* category_data, uint8_t top_category_index) {
#ifdef CONFIG_MICRO_KWS_PRINT_OUTPUTS
  for (size_t i = 0; i < category_count; i++) {
    // Convert top category name to UPPER CASE.
    if (i == top_category_index) {
      // Since C has no function for strings, we need to iterate over all chars.
      for (size_t j = 0; category_labels[i][j] != '\0'; j++)
        printf("%c", toupper(category_labels[i][j]));
      printf(":%4d  ", category_data[i]);
    } else
      printf("%s:%4d  ", category_labels[i], category_data[i]);
  }
#endif  // CONFIG_MICRO_KWS_PRINT_OUTPUTS

#ifdef CONFIG_MICRO_KWS_PRINT_TIME
  static uint32_t last_time = (uint32_t)(esp_timer_get_time() / 1000);
  uint32_t this_time = (uint32_t)(esp_timer_get_time() / 1000);
  int32_t delta_time = this_time - last_time;
#ifdef CONFIG_MICRO_KWS_PRINT_OUTPUTS
  printf("\t");
#endif  // CONFIG_MICRO_KWS_PRINT_OUTPUTS
  printf("Δ%" PRId32 "ms", delta_time);
  last_time = this_time;
#endif  // CONFIG_MICRO_KWS_PRINT_TIME
#ifdef CONFIG_MICRO_KWS_MODE_DEFAULT
  printf("\n");
#endif  // CONFIG_MICRO_KWS_MODE_DEFAULT

#ifdef CONFIG_MICRO_KWS_MODE_DEBUG
  debug_data_t debug_data;
  memcpy(debug_data.feature_data, feature_data, feature_element_count);
  memcpy(debug_data.category_data, category_data, category_count);
  debug_data.top_category_index = top_category_index;

  if (xRingbufferSend(buf_handle, (void*)&debug_data, sizeof(debug_data), pdMS_TO_TICKS(100)) !=
      pdTRUE) {
    ESP_LOGE(__FILE__,
             "ERROR: In xRingbufferSend() in DebugRun(). Most "
             "likely the Ringbuffer is full. Make sure the DebugWorker() is "
             "running and reading enough to keep the Ringbuffer empty.");
    return ESP_FAIL;
  }
#endif  // CONFIG_MICRO_KWS_MODE_DEBUG

  return ESP_OK;
}

#else   // CONFIG_MICRO_KWS_MODE_DEBUG_AUDIO

esp_err_t DebugRunAudio(int8_t* audio_data) {
  // TODO(fabianpedd): Switch these over to 32bit
  static uint32_t last_time = (uint32_t)(esp_timer_get_time() / 1000);
  uint32_t this_time = (uint32_t)(esp_timer_get_time() / 1000);
  int32_t delta_time = this_time - last_time;
  printf("Δ%dms", delta_time);
  last_time = this_time;
  printf("\n");

  debug_data_t debug_data;
  memcpy(debug_data.audio_data, audio_data, AUDIO_PACKET_SIZE);

  if (xRingbufferSend(buf_handle, (void*)&debug_data, sizeof(debug_data), pdMS_TO_TICKS(500)) !=
      pdTRUE) {
    ESP_LOGE(__FILE__,
             "ERROR: In xRingbufferSend() in DebugRun(). Most "
             "likely the Ringbuffer is full.");
    return ESP_FAIL;
  }
  printf("Sent %d bytes via xRingbufferSend() in DebugRunAudio().\n", sizeof(debug_data));

  return ESP_OK;
}
#endif  // CONFIG_MICRO_KWS_MODE_DEBUG_AUDIO

void micro_audio(void* params) {
  printf("Starting audio recording task...\n");

  // Initialize onboard LEDs, if available.
  if (InitializeGPIO() != ESP_OK) {
    ESP_LOGE(__FILE__, "ERROR: In InitializeGPIO().");
    return;
  }

  // Set RGB to yellow in order to indicate standby.
  SetLEDColor(LED_RGB_YELLOW);

  // Wait some amount of time before proceeding in order to let speaker get
  // ready.
  vTaskDelay(pdMS_TO_TICKS(2000));

  // Only now initalize audio driver.
  if (InitializeAudio() != ESP_OK) {
    ESP_LOGE(__FILE__, "ERROR: In InitializeAudio().");
    return;
  }

  // Set RGB to orange in order to indicate get ready
  SetLEDColor(LED_RGB_ORANGE);

  // Large array that holds the complete audio sample.
  int8_t i2s_read_buffer[AUDIO_SAMPLE_SIZE] = {0};
  size_t actual_bytes_read = 0;

  // Discard 1 second of audio data in order to let audio level stabilize.
  uint32_t start_time = (uint32_t)(esp_timer_get_time() / 1000);
  do {
    vTaskDelay(pdMS_TO_TICKS(10));
    if (GetAudioData(2 * 16 * 200, &actual_bytes_read, i2s_read_buffer) != ESP_OK) {
      ESP_LOGE(__FILE__, "ERROR: In GetAudioData().");
      return;
    } else if (actual_bytes_read > 0) {
      printf("Discarding %d bytes...\n", actual_bytes_read);
    }
  } while (actual_bytes_read == 0 || (uint32_t)(esp_timer_get_time() / 1000) - start_time < 1000);

  // Set RGB to red in order to indicate start of recording.
  SetLEDColor(LED_RGB_RED);
  printf("Starting recording at %" PRId32 " ms...\n", (uint32_t)(esp_timer_get_time() / 1000) - start_time);

  // Start actual recording until all data is collected.
  size_t total_bytes_read = 0;
  actual_bytes_read = 0;
  for (;;) {
    if (GetAudioData(MIN(AUDIO_PACKET_SIZE, AUDIO_SAMPLE_SIZE - total_bytes_read),
                     &actual_bytes_read, &(i2s_read_buffer[total_bytes_read])) != ESP_OK) {
      ESP_LOGE(__FILE__, "ERROR: In GetAudioData().");
      return;
    }

    if (actual_bytes_read > AUDIO_PACKET_SIZE) {
      ESP_LOGE(__FILE__,
               "ERROR: actual_bytes_read %d from GetAudioData() greater than "
               "AUDIO_PACKET_SIZE %d.",
               actual_bytes_read, AUDIO_PACKET_SIZE);
      return;
    }

    total_bytes_read += actual_bytes_read;
    if (total_bytes_read == AUDIO_SAMPLE_SIZE) {
      break;
    } else if (total_bytes_read > AUDIO_SAMPLE_SIZE) {
      ESP_LOGE(__FILE__,
               "ERROR: total_bytes_read %d from GetAudioData() greater than "
               "AUDIO_SAMPLE_SIZE %d.",
               total_bytes_read, AUDIO_SAMPLE_SIZE);
      return;
    }

    vTaskDelay(pdMS_TO_TICKS(10));
  }

  // Stop the I2S audio driver.
  StopAudio();

  // Set RGB to yellow in order to indicate transmission to host PC.
  SetLEDColor(LED_RGB_YELLOW);
  printf("Stopping recording at %" PRId32 " ms.\n", (uint32_t)(esp_timer_get_time() / 1000) - start_time);

  // Only now initalize UART Debug driver.
  if (InitializeDebug() != ESP_OK) {
    ESP_LOGE(__FILE__, "ERROR: In InitializeDebug().");
    return;
  }

  // Give the UART some to settle down.
  vTaskDelay(pdMS_TO_TICKS(100));

  // TODO(fabianpedd): Maybe send first packet twice in order to prevent inital
  // data loss?
  // DebugRunAudio(&(i2s_read_buffer[0]));
  // vTaskDelay(pdMS_TO_TICKS(500));

  // Transmit UART packages with enough time inbetween.
  for (size_t i = 0; i < total_bytes_read; i += AUDIO_PACKET_SIZE) {
    DebugRunAudio(&(i2s_read_buffer[i]));
    vTaskDelay(pdMS_TO_TICKS(500));
  }

  // Stop the UART Debug driver and indicate completion of transmission.
  StopDebug();
  SetLEDColor(LED_RGB_GREEN);

  printf("Successfully recorded and transmitted audio.\n");

  // Wait forever...
  while (true) {
    vTaskDelay(pdMS_TO_TICKS(100));
  }
}
