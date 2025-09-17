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

#ifndef GPIO_H
#define GPIO_H

#include <cstdint>

#include "driver/gpio.h"
#include "esp_err.h"
#include "sdkconfig.h"

// TODO(fabianpedd): We need to check that we are not interfering with other
// functions on the gpio pins below, like UART or SPI Flash.

// Define some handy LED colors that can be passed to SetLEDColor
#define LED_RGB_BLACK 0, 0, 0
#define LED_RGB_WHITE 255, 255, 255
#define LED_RGB_RED 255, 0, 0
#define LED_RGB_GREEN 0, 255, 0
#define LED_RGB_BLUE 0, 0, 255
#define LED_RGB_YELLOW 255, 255, 0
#define LED_RGB_CYAN 0, 255, 255
#define LED_RGB_MAGENTA 255, 0, 255
#define LED_RGB_ORANGE 255, 127, 0
#define LED_RGB_PURPLE 127, 0, 255
#define LED_RGB_MINT 0, 255, 127

#if CONFIG_IDF_TARGET_ESP32C3
#pragma message("Setting GPIO Pins for RISC-V ESP32-C3")

#define I2S_SCK_PIN 7
#define I2S_WS_PIN 6
#define I2S_DATA_IN_PIN 8
#define I2S_PORT_ID 0

#define GPIO_LED_STATUS_A ((gpio_num_t)19)  // white status led
#define GPIO_LED_STATUS_B ((gpio_num_t)18)  // orange status led
#define GPIO_LED_RED ((gpio_num_t)3)
#define GPIO_LED_GREEN ((gpio_num_t)4)
#define GPIO_LED_BLUE ((gpio_num_t)5)

#elif CONFIG_IDF_TARGET_ESP32
#warning "Setting GPIO Pins for Xtensa ESP32"

#define I2S_SCK_PIN 32
#define I2S_WS_PIN 25
#define I2S_DATA_IN_PIN 33
#define I2S_PORT_ID 1

#else

#error "ESP-IDF target not supported. Please provide information manually in pin_def.h"

#endif

esp_err_t InitializeGPIO(void);

esp_err_t SetLEDColor(uint8_t red, uint8_t green, uint8_t blue);

#endif  // GPIO_H
