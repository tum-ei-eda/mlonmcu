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

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "audio.h"
#include "backend.h"
#include "debug.h"
#include "driver/i2s.h"
#include "driver/uart.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "frontend.h"
#include "gpio.h"
#include "model_settings.h"
// #include "tvm_wrapper.h"
#include "ml_interface.h"

#ifndef CONFIG_ENABLE_WIFI
#define CONFIG_ENABLE_WIFI 0
#endif

#if CONFIG_ENABLE_WIFI
#include "wifi_udp.h"
#include "nvs_flash.h"
#endif //CONFIG_ENABLE_WIFI

#ifndef ENABLE_PERF_EVAL
#define ENABLE_PERF_EVAL 0
#endif
#if ENABLE_PERF_EVAL
#include "dummy_audio.h"
#include "bench.h"
#endif //ENABLE_PERF_EVAL

// TODO(fabianpedd): Use size_t wherever reasonable
// TODO(fabianpedd): Adjust return values and ESP_LOG to esp-idf specific types
// and functions

#if CONFIG_ENABLE_WIFI
#define MAX_KEYWORD_LEN 16
static struct __attribute__((packed, aligned(4))) {
  char detected_keyword[MAX_KEYWORD_LEN];
} udp_msg;
const char* keyword;
#endif // CONFIG_ENABLE_WIFI

void micro_kws(void* params) {
#if ENABLE_PERF_EVAL
  target_init();
  start_bench(TOTAL);
  start_bench(INIT);
#endif //ENABLE_PERF_EVAL
  // Initialize onboard LEDs, if available.
  if (InitializeGPIO() != ESP_OK) {
    ESP_LOGE(__FILE__, "ERROR: In InitializeGPIO().");
    return;
  }
#if !ENABLE_PERF_EVAL
  if (InitializeAudio() != ESP_OK) {
    ESP_LOGE(__FILE__, "ERROR: In InitializeAudio().");
    return;
  }
#endif //ENABLE_PERF_EVAL

  if (InitializeFrontend() != ESP_OK) {
    ESP_LOGE(__FILE__, "ERROR: In InitializeFrontend().");
    return;
  }

  // This is only relevant when using the Python visualizer via the additional
  // UART interface.
  if (InitializeDebug() != ESP_OK) {
    ESP_LOGE(__FILE__, "ERROR: In InitializeDebug().");
    return;
  }

  // We are collecting 20ms of new audio data and are reusing 10ms of past data.
  // time    30ms = 20ms + 10ms
  // samples 480 = 320 new + 160 old
  // bytes   960 = 640 new + 320 old
  int8_t audio_buffer[960] = {0};

  // Contains our features. Interpreted as 40 by 49 byte 2d array.
  int8_t feature_buffer[feature_element_count];

  // Endless loop of main function.
  printf("Starting system main loop...\n");

  TickType_t last_inference_ticks = xTaskGetTickCount();
  ;
  const TickType_t min_inference_ticks = (1000 / CONFIG_MICRO_KWS_MAX_RATE) / portTICK_PERIOD_MS;

  mlif_init();
#if ENABLE_PERF_EVAL
  size_t run_cnt = 0;
  stop_bench(INIT);
  while (DummyAudioRemaining()) {
#else //ENABLE_PERF_EVAL
  while (true) {
#endif //ENABLE_PERF_EVAL
    // Get audio data from audio input and create slices until no more data is
    // available. But at most `feature_slize_count` times, which is equal to
    // 960ms of data. If we would
    for (size_t i = 0; i < feature_slize_count; i++) {
      // Get audio data via I2S from the audio driver.
      size_t actual_bytes_read = 0;
      int8_t i2s_read_buffer[640] = {0};
#if ENABLE_PERF_EVAL
      if (DummyAudioRemaining()) {
        GetDummyAudioData(640, &actual_bytes_read, i2s_read_buffer);
      }
      else {
        break;
      }
#else //ENABLE_PERF_EVAL
      if (GetAudioData(640, &actual_bytes_read, i2s_read_buffer) != ESP_OK) {
        ESP_LOGE(__FILE__, "ERROR: In GetAudioData().");
        return;
      }
#endif //ENABLE_PERF_EVAL

      // If there is no more audio data available at the moment, exit the
      // loop and continue with inference.
      if (actual_bytes_read < 640) {
        break;
      }

      // If there is a full 20ms / 320 samples / 640 bytes available, move the
      // old data (10ms / 160 samples / 320 bytes) to the top of the buffer and
      // fill the rest with the new audio data from the i2s_read_buffer.
      memcpy(audio_buffer, audio_buffer + 640, 320);
      memcpy(audio_buffer, i2s_read_buffer, 640);

      // Generate a new feature slice from the audio samples using the
      // GenerateFrontendData() function. This will convert the time domain
      // audio samples into a frequency domain representation.
      int8_t new_slice_buffer[feature_slice_size] = {0};
      if (GenerateFrontendData((int16_t*)audio_buffer, 512, new_slice_buffer) != 0) {
        ESP_LOGE(__FILE__, "ERROR: In GenerateFrontendData().");
        return;
      }

      // Move other slices by one, i.e. make room to store new slice at the end.
      // TODO(fabianpedd): This is actually really inefficient. Using a
      // ringbuffer would be a lot more efficient but also more
      // complicated. The ringbuffer could be used in such a way as to
      // minimize the amount of copying required. It would only be
      // necessary to copy the data when a ringbuffer overflow occurs. The
      // ringbuffer, in turn, would have to be at least 2-3x times the size of
      // the data we are expecting in order for this method to bring any
      // improvement. So we are basically trading in storage (of which we should
      // have plenty) for computations. But then again, how expensive are a
      // couple of memmoves and memcpys in the grand scheme of things here?
      memmove(feature_buffer, feature_buffer + feature_slice_size,
              feature_element_count - feature_slice_size);
      memcpy(feature_buffer + feature_element_count - feature_slice_size, new_slice_buffer,
             feature_slice_size);
    }

    // Copy the feature buffer into the model input buffer and run the
    // inference.
    memcpy((int8_t*)mlif_input_ptr(0), feature_buffer, feature_element_count);

    // Limit number of inferences per second
    vTaskDelayUntil(&last_inference_ticks, min_inference_ticks);

#if ENABLE_PERF_EVAL
    if(run_cnt == 0) { 
      //vTaskSuspendAll();
      taskENTER_CRITICAL(0);
      start_bench(RUN); 
    }
#endif // ENABLE_PERF_EVAL

    mlif_invoke();

#if ENABLE_PERF_EVAL
    if(run_cnt == 0) { 
      stop_bench(RUN); 
      taskEXIT_CRITICAL(0);
      //xTaskResumeAll();
    }
    run_cnt++;
#endif // ENABLE_PERF_EVAL
    // Collect and offest the inference values by 128
    uint8_t output[category_count] = {0};
    for (size_t i = 0; i < category_count; i++) {
      output[i] = ((int8_t*)mlif_output_ptr(0))[i] + 128;
    }

    size_t top_category_index = 0;

#ifdef CONFIG_MICRO_KWS_LED_RAW_POSTERIORS
    SetLEDColor(output[3], output[2], 0);
#else   // CONFIG_MICRO_KWS_LED_RAW_POSTERIORS
    HandlePosteriors(output, &top_category_index);
#endif  // CONFIG_MICRO_KWS_LED_RAW_POSTERIORS

#if CONFIG_ENABLE_WIFI
    keyword = category_labels[top_category_index];
    strncpy(udp_msg.detected_keyword,keyword,MAX_KEYWORD_LEN-1);
    udp_msg.detected_keyword[MAX_KEYWORD_LEN-1] = '\0';
    ESP_LOGI("Main", "Detected keyword: %s", udp_msg.detected_keyword);
    udp_send(&udp_msg, MAX_KEYWORD_LEN);
#endif //CONFIG_ENABLE_WIFI
    // Send the feature buffer and inferences results to the computer for
    // analysis and debugging.
    DebugRun(feature_buffer, output, top_category_index);
  }
#if ENABLE_PERF_EVAL
  stop_bench(TOTAL);
  print_bench(INIT);
  print_bench(RUN);
  print_bench(TOTAL);
  mlonmcu_printf("Program finish.\n");
  target_deinit();
#endif //ENABLE_PERF_EVAL
}

// This function gets called after the internal bootprocess is finished. We
// directly span a new task in which we do all the work.
extern "C" void app_main(void) {
#if ENABLE_PERF_EVAL
  xTaskCreate(&micro_kws,   // Function that implements the task.
              "micro_kws",  // Text name for the task.
              32 * 1024,    // Stack size in bytes, so 32KB.
              NULL,         // Parameter passed into the task.
              8,            // Priority of task. Higher number, higher prio.
              NULL          // Task handle.
  );

#else //ENABLE_PERF_EVAL

#if CONFIG_ENABLE_WIFI
  // Initialize NVS
  esp_err_t ret = nvs_flash_init();
  if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
      ESP_ERROR_CHECK(nvs_flash_erase());
      ret = nvs_flash_init();
  }
  ESP_ERROR_CHECK(ret);

  // Initialize Wi-Fi
  ESP_LOGI("WiFi", "ESP_WIFI_MODE_STA");
  wifi_scan();
  wifi_init();
#endif //CONFIG_ENABLE_WIFI

#ifndef CONFIG_MICRO_KWS_MODE_DEBUG_AUDIO
  xTaskCreate(&micro_kws,   // Function that implements the task.
              "micro_kws",  // Text name for the task.
              32 * 1024,    // Stack size in bytes, so 32KB.
              NULL,         // Parameter passed into the task.
              8,            // Priority of task. Higher number, higher prio.
              NULL          // Task handle.
  );
#else   // CONFIG_MICRO_KWS_MODE_DEBUG_AUDIO
  xTaskCreate(&micro_audio,   // Function that implements the task.
              "micro_audio",  // Text name for the task.
              96 * 1024,      // Stack size in bytes.
              NULL,           // Parameter passed into the task.
              8,              // Priority of task. Higher number, higher prio.
              NULL            // Task handle.
  );
#endif  // CONFIG_MICRO_KWS_MODE_DEBUG_AUDIO

#endif //ENABLE_PERF_EVAL
}
