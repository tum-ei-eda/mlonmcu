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

#include "backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>

#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "gpio.h"
#include "model_settings.h"
#include "posterior.h"

// TODO(fabianpedd): In order to improve the detection accuracy we could
// introduce a rejection difference threshold. The difference between the top
// category and the 2nd top category would have to be higher than this
// threshold. If smaller, we would simply categorize silence or unknown. This
// would help to suppress detections where the NN is unsure about two categories.
// However, this might lead to worse detection sensitivity.

// TODO(fabianpedd): Make threshold (or better history length) independent from
// inference performance / inferences per second.
constexpr size_t posterior_suppression_ms = CONFIG_MICRO_KWS_POSTERIOR_SUPPRESSION_MS;
constexpr size_t posterior_history_length = CONFIG_MICRO_KWS_POSTERIOR_HISTORY_LENGTH;
constexpr size_t posterior_trigger_threshold = CONFIG_MICRO_KWS_POSTERIOR_TRIGGER_THRESHOLD_SINGLE;
constexpr size_t posterior_category_count = CONFIG_MICRO_KWS_NUM_CLASSES;

/**
 * @brief Called if a new category was detected to handle the LEDs and Console output.
 *
 * @param category The detected category/label passed as a char string.
 *
 * @return ESP_OK if no error occured.
 *
 */
esp_err_t KeywordCallback(const char* category) {

  // print message to serial console
  printf("Detected a new keyword %s\n", category);

  // handle LEDs
  if (strcmp(category, CONFIG_MICRO_KWS_CLASS_LABEL_0) == 0) {
    SetLEDColor(LED_RGB_BLACK);  // silence -> off
  } else if (strcmp(category, CONFIG_MICRO_KWS_CLASS_LABEL_1) == 0) {
    SetLEDColor(LED_RGB_ORANGE);  // unknown -> orange
  } else if (strcmp(category, CONFIG_MICRO_KWS_CLASS_LABEL_2) == 0) {
    SetLEDColor(LED_RGB_GREEN);  // yes -> green
  } else if (strcmp(category, CONFIG_MICRO_KWS_CLASS_LABEL_3) == 0) {
    SetLEDColor(LED_RGB_RED);  // no -> red
#ifdef CONFIG_MICRO_KWS_CLASS_LABEL_4
  } else if (strcmp(category, CONFIG_MICRO_KWS_CLASS_LABEL_4) == 0) {
    SetLEDColor(LED_RGB_BLUE);
#endif  // CONFIG_MICRO_KWS_CLASS_LABEL_4
#ifdef CONFIG_MICRO_KWS_CLASS_LABEL_5
  } else if (strcmp(category, CONFIG_MICRO_KWS_CLASS_LABEL_5) == 0) {
    SetLEDColor(LED_RGB_YELLOW);
#endif  // CONFIG_MICRO_KWS_CLASS_LABEL_5
#ifdef CONFIG_MICRO_KWS_CLASS_LABEL_6
  } else if (strcmp(category, CONFIG_MICRO_KWS_CLASS_LABEL_6) == 0) {
    SetLEDColor(LED_RGB_CYAN);
#endif  // CONFIG_MICRO_KWS_CLASS_LABEL_6
#ifdef CONFIG_MICRO_KWS_CLASS_LABEL_7
  } else if (strcmp(category, CONFIG_MICRO_KWS_CLASS_LABEL_7) == 0) {
    SetLEDColor(LED_RGB_MAGENTA);
#endif  // CONFIG_MICRO_KWS_CLASS_LABEL_7
#ifdef CONFIG_MICRO_KWS_CLASS_LABEL_8
  } else if (strcmp(category, CONFIG_MICRO_KWS_CLASS_LABEL_8) == 0) {
    SetLEDColor(LED_RGB_PURPLE);
#endif  // CONFIG_MICRO_KWS_CLASS_LABEL_8
#ifdef CONFIG_MICRO_KWS_CLASS_LABEL_9
  } else if (strcmp(category, CONFIG_MICRO_KWS_CLASS_LABEL_9) == 0) {
    SetLEDColor(LED_RGB_MINT);
#endif  // CONFIG_MICRO_KWS_CLASS_LABEL_9
  } else {
    SetLEDColor(LED_RGB_BLACK);  // invalid -> off
    return ESP_FAIL;
  }

  return ESP_OK;
}

/**
 * @brief Wrapper calling the posterior handler after each inference and responding to
 * classifications.
 *
 * @param new_posteriors The raw model outputs with unsigned 8-bit values.
 * @param top_category_index The index of the detected category/label returned by reference.
 *
 * @return ESP_OK if no error occured.
 *
 */
esp_err_t HandlePosteriors(uint8_t new_posteriors[category_count], size_t* top_category_index) {
  // Create a single instance with infinite lifetime (no cleanup required)
  static PosteriorHandler* handler =
      new PosteriorHandler(posterior_history_length, posterior_trigger_threshold,
                           posterior_suppression_ms, posterior_category_count);

  // local variables
  esp_err_t ret;
  bool trigger = false;

  // convert us into ms
  uint32_t time_ms = esp_timer_get_time() / 1000;

  // call posterior handler (to be implemented by students)
  ret = handler->Handle(new_posteriors, time_ms, top_category_index, &trigger);

  // if a keyword was detected, turn on the LEDs and print a message to the console
  if (trigger) {
    KeywordCallback(category_labels[*top_category_index]);
  }

  return ret;
}
