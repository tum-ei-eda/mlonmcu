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

#include "gpio.h"

#include "driver/gpio.h"
#include "driver/ledc.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define LED_RED_CHANNEL LEDC_CHANNEL_0
#define LED_GREEN_CHANNEL LEDC_CHANNEL_1
#define LED_BLUE_CHANNEL LEDC_CHANNEL_2

esp_err_t InitializeGPIO() {
#ifdef GPIO_LED_STATUS_A
  gpio_reset_pin(GPIO_LED_STATUS_A);
  gpio_set_direction(GPIO_LED_STATUS_A, GPIO_MODE_OUTPUT);
#endif
#ifdef GPIO_LED_STATUS_B
  gpio_reset_pin(GPIO_LED_STATUS_B);
  gpio_set_direction(GPIO_LED_STATUS_B, GPIO_MODE_OUTPUT);
#endif
#ifdef GPIO_LED_RED
  gpio_reset_pin(GPIO_LED_RED);
  gpio_set_direction(GPIO_LED_RED, GPIO_MODE_OUTPUT);
#endif
#ifdef GPIO_LED_GREEN
  gpio_reset_pin(GPIO_LED_GREEN);
  gpio_set_direction(GPIO_LED_GREEN, GPIO_MODE_OUTPUT);
#endif
#ifdef GPIO_LED_BLUE
  gpio_reset_pin(GPIO_LED_BLUE);
  gpio_set_direction(GPIO_LED_BLUE, GPIO_MODE_OUTPUT);
#endif

  ledc_timer_config_t ledc_timer = {.speed_mode = LEDC_LOW_SPEED_MODE,
                                    .duty_resolution = LEDC_TIMER_8_BIT,
                                    .timer_num = LEDC_TIMER_0,
                                    .freq_hz = 1000,
                                    .clk_cfg = LEDC_AUTO_CLK};
  ledc_channel_config_t ledc_channel = {.gpio_num = GPIO_LED_RED,
                                        .speed_mode = LEDC_LOW_SPEED_MODE,
                                        .channel = LED_RED_CHANNEL,
                                        .intr_type = LEDC_INTR_DISABLE,
                                        .timer_sel = LEDC_TIMER_0,
                                        .duty = 0,
                                        .hpoint = 0/*,
                                        .flags.output_invert = 0*/}; // TODO(fabianpedd): For whatever reason the compiler is not happy about this?!
  ledc_timer_config(&ledc_timer);
  ledc_channel_config(&ledc_channel);

  ledc_channel.gpio_num = GPIO_LED_GREEN;
  ledc_channel.channel = LED_GREEN_CHANNEL;
  ledc_timer_config(&ledc_timer);
  ledc_channel_config(&ledc_channel);

  ledc_channel.gpio_num = GPIO_LED_BLUE;
  ledc_channel.channel = LED_BLUE_CHANNEL;
  ledc_timer_config(&ledc_timer);
  ledc_channel_config(&ledc_channel);

  return ESP_OK;
}

esp_err_t SetLEDColor(uint8_t red, uint8_t green, uint8_t blue) {
  ledc_set_duty(LEDC_LOW_SPEED_MODE, LED_RED_CHANNEL, red);
  ledc_update_duty(LEDC_LOW_SPEED_MODE, LED_RED_CHANNEL);
  ledc_set_duty(LEDC_LOW_SPEED_MODE, LED_GREEN_CHANNEL, green);
  ledc_update_duty(LEDC_LOW_SPEED_MODE, LED_GREEN_CHANNEL);
  ledc_set_duty(LEDC_LOW_SPEED_MODE, LED_BLUE_CHANNEL, blue);
  ledc_update_duty(LEDC_LOW_SPEED_MODE, LED_BLUE_CHANNEL);

  return ESP_OK;
}
