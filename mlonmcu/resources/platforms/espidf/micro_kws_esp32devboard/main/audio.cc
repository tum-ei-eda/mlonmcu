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

#include "audio.h"

#include <cstring>

#include "driver/i2s.h"
#include "esp_log.h"
#include "esp_spi_flash.h"
#include "freertos/ringbuf.h"
#include "gpio.h"

static RingbufHandle_t buf_handle = NULL;
static TaskHandle_t CaptureAudioSamplesHandle = NULL;

static void CaptureAudioSamples(void* arg) {
  constexpr size_t bytes_to_read = 512;
  size_t bytes_read = 0;
  int8_t data_buf[bytes_to_read] = {0};

  while (1) {
    i2s_read((i2s_port_t)I2S_PORT_ID, (void*)data_buf, (size_t)bytes_to_read, (size_t*)&bytes_read,
             pdMS_TO_TICKS(100));

    if (bytes_read < bytes_to_read) {
      ESP_LOGE(__FILE__, "ERROR: In i2s_read(). Could ony read %d of %d bytes.", bytes_read,
               bytes_to_read);
      return;
    }

    if (xRingbufferSend(buf_handle, data_buf, bytes_read, pdMS_TO_TICKS(100)) != pdTRUE) {
      ESP_LOGE(__FILE__, "ERROR: In xRingbufferSend(). Could not send %d bytes.", bytes_read);
      return;
    }
  }
}

esp_err_t InitializeAudio() {
  i2s_config_t i2s_config = {
      /* Master should supply clock and we are only receiving data. */
      .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
      /* Sample rate of 16KHz */
      .sample_rate = 16000,
      /* 16bit per sample, i.e. two bytes. */
      .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
      /* We only have a mono microphone which is outputting its audio data into
         the left channel of the I2S interface (L/R pin connected to GND). Thus,
         we are only interested in reading the left channel of the I2S
         interface. */
      .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
      /* We are using a standard I2S interface. */
      .communication_format = I2S_COMM_FORMAT_STAND_I2S,
      /* Interrupt level set to 1 for the I2S hardware interrupt. */
      .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
      /* Using 3 internal buffers with 300 samples each. */
      .dma_buf_count = 3,
      .dma_buf_len = 300,
      /* No need for the higher resolution APLL clock. */
      .use_apll = false,
      /* No need for the auto clear feature since we are not transmitting. */
      .tx_desc_auto_clear = false,
  };

  i2s_pin_config_t pin_config = {
      /* No master clock needed. We are "only" using the bit clock. */
      .mck_io_num = I2S_PIN_NO_CHANGE,
      .bck_io_num = I2S_SCK_PIN,
      /* Word select line. Selects the left or right channel of the slave
         device. */
      .ws_io_num = I2S_WS_PIN,
      /* Data out is not needed as we are only working with a "data source". */
      .data_out_num = I2S_PIN_NO_CHANGE,
      /* Data in from slave device. */
      .data_in_num = I2S_DATA_IN_PIN};

  esp_err_t ret = ESP_OK;

  ret = i2s_driver_install((i2s_port_t)I2S_PORT_ID, &i2s_config, 0, NULL);
  if (ret != ESP_OK) {
    ESP_LOGE(__FILE__, "ERROR: In InitializeAudio() at i2s_driver_install().");
    return ret;
  }

  ret = i2s_set_pin((i2s_port_t)I2S_PORT_ID, &pin_config);
  if (ret != ESP_OK) {
    ESP_LOGE(__FILE__, "ERROR: In InitializeAudio() at i2s_set_pin().");
    return ret;
  }

  ret = i2s_zero_dma_buffer((i2s_port_t)I2S_PORT_ID);
  if (ret != ESP_OK) {
    ESP_LOGE(__FILE__, "ERROR: In InitializeAudio() at i2s_zero_dma_buffer().");
    return ret;
  }

  buf_handle = xRingbufferCreate(1024 * 32, RINGBUF_TYPE_BYTEBUF);
  if (buf_handle == NULL) {
    ESP_LOGE(__FILE__, "ERROR: In InitializeAudio() at xRingbufferCreate().");
    return ret;
  }

  if (xTaskCreate(CaptureAudioSamples, "CaptureAudioSamples", 1024 * 32, NULL, 10,
                  &CaptureAudioSamplesHandle) != pdPASS) {
    ESP_LOGE(__FILE__, "ERROR: In InitializeAudio() at xTaskCreate(CaptureAudioSamples).");
    return ESP_FAIL;
  }

  return ret;
}

esp_err_t StopAudio() {
  // TODO(fabianpedd): Also free Ringbuffer and deinstall I2S driver
  vTaskDelete(CaptureAudioSamplesHandle);
  return ESP_OK;
}

esp_err_t GetAudioData(size_t requested_size, size_t* actual_size, int8_t* data) {
  // Set returned number of bytes to zero for now.
  *actual_size = 0;

  // Peak into the Ringbuffer.
  size_t bytes_waiting = 0;
  vRingbufferGetInfo(buf_handle, NULL, NULL, NULL, NULL, (UBaseType_t*)&bytes_waiting);

  // Check if we actually have the requested amount of bytes available. If yes,
  // get the data. If not, simply return zero.
  if (bytes_waiting >= requested_size) {
    // Get the data from the Ringbuffer.
    size_t bytes_received = 0;
    int8_t* buf_data = (int8_t*)xRingbufferReceiveUpTo(buf_handle, (size_t*)&bytes_received,
                                                       pdMS_TO_TICKS(100), requested_size);

    // Check whether we have encountered a wraparound in the Ringbuffer. If so,
    // we need to read a second time in order to retrieve all data. See here
    // https://docs.espressif.com/projects/esp-idf/en/v4.4/esp32c3/api-reference/system/freertos_additions.html#_CPPv422xRingbufferReceiveUpTo15RingbufHandle_tP6size_t10TickType_t6size_t
    if (buf_data != NULL && bytes_received < requested_size) {
      // Copy the data from the Ringbuffer into output buffer and free the
      // ringbuffer data.
      memcpy(data, buf_data, bytes_received);
      vRingbufferReturnItem(buf_handle, (void*)buf_data);

      // Move the data pointer and adjust the amount of data needed accordingly.
      data += bytes_received;
      *actual_size += bytes_received;
      requested_size -= bytes_received;

      // Get the rest of the data from the top of the Ringbuffer.
      buf_data = (int8_t*)xRingbufferReceiveUpTo(buf_handle, (size_t*)&bytes_received,
                                                 pdMS_TO_TICKS(100), requested_size);
    }

    if (buf_data != NULL && bytes_received == requested_size) {
      // Copy the data from the Ringbuffer into output buffer and free the
      // ringbuffer data.
      memcpy(data, buf_data, bytes_received);
      vRingbufferReturnItem(buf_handle, (void*)buf_data);

      *actual_size += bytes_received;
      return ESP_OK;

    } else {
      ESP_LOGE(__FILE__,
               "ERROR: Only read %d of %d bytes from Ringbuffer. Something went "
               "wrong, as there should be enough data available.",
               *actual_size, requested_size);
      // TODO(fabianpedd): Should not be needed here. But maybe reintroduce as a
      // safety measure?
      // vRingbufferReturnItem(buf_handle, (void *)buf_data);
      return ESP_FAIL;
    }
  } else {
    // Did not have enough data currently. No big deal, so print no error.
    data = NULL;
    return ESP_OK;
  }
}
