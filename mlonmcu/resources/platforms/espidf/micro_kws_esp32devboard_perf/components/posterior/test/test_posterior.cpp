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

#include <limits.h>
#include "unity.h"
#include "posterior.h"

TEST_CASE("History length: 35, Threshold: 100, Suppression: 0 ms", "[default]") {
  // constants
  constexpr uint32_t history_length = 35;
  constexpr uint8_t trigger_threshold_single = 100;
  constexpr uint32_t suppression_ms = 0;
  constexpr uint32_t category_count = 4;

  // local working variables
  esp_err_t ret = ESP_OK;
  bool trigger = false;
  size_t trigger_count = 0;
  size_t top_category_index = 0;
  uint8_t new_posteriors[4] = {0};

  // dummy time for deterministic results
  constexpr uint32_t time_delta = 10;
  uint32_t fake_time = 0;

  // create instance of posterior handler to be tested
  PosteriorHandler *handler = new PosteriorHandler(history_length, trigger_threshold_single,
                                                   suppression_ms, category_count);

  // Fill with Silence (first 35 iterations have undefined behavior)
  new_posteriors[0] = 255;  // silence
  new_posteriors[1] = 0;    // unknown
  new_posteriors[2] = 0;    // yes
  new_posteriors[3] = 0;    // no
  trigger_count = 0;
  for (size_t i = 0; i < history_length; i++) {
    // update time and reset trigger
    fake_time += time_delta;
    trigger = false;

    // invoke posterior handler
    ret = handler->Handle(new_posteriors, fake_time, &top_category_index, &trigger);

    // check if successful
    TEST_ASSERT_EQUAL(ESP_OK, ret);

    // comment in for debugging
    // printf("i=%u top=%u trigger=%x count=%u\n", i, top_category_index, trigger, trigger_count);

    // handle outputs
    if (trigger) {
      trigger_count++;
    }
  }
  // basic assertions (not very strict because first outputs may not be well-defined)
  TEST_ASSERT_EQUAL(0, top_category_index);
  TEST_ASSERT_TRUE(trigger_count > 0);

  // Fill with Unknown
  new_posteriors[0] = 0;    // silence
  new_posteriors[1] = 255;  // unknown
  new_posteriors[2] = 0;    // yes
  new_posteriors[3] = 0;    // no
  trigger_count = 0;
  for (size_t i = 0; i < history_length; i++) {
    // update time and reset trigger
    fake_time += time_delta;
    trigger = false;

    // invoke posterior handler
    ret = handler->Handle(new_posteriors, fake_time, &top_category_index, &trigger);

    // check if successful
    TEST_ASSERT_EQUAL(ESP_OK, ret);

    // comment in for debugging
    // printf("i=%u top=%u trigger=%x count=%u\n", i, top_category_index, trigger, trigger_count);

    // handle outputs
    if (trigger) {
      trigger_count++;
    }

    // after 17+-1 interations the label should schange from silence to unknown
    if (i < 16) {
      TEST_ASSERT_EQUAL(0, top_category_index);  // silence
    } else if (i >= 16 && i <= 18) {
      // tol
    } else if (i > 18) {
      TEST_ASSERT_EQUAL(1, top_category_index);  // unknown
    } else {
      // should never be reached
    }
  }

  // further assertions (suppression_ms=0 -> continuous triggering)
  TEST_ASSERT_EQUAL(history_length, trigger_count);
  TEST_ASSERT_EQUAL(1, top_category_index);  // unknown

  // Fill with Mixed outputs
  new_posteriors[0] = 16;   // yes
  new_posteriors[1] = 32;   // unknown
  new_posteriors[2] = 144;  // yes
  new_posteriors[3] = 64;   // no
  trigger_count = 0;
  for (size_t i = 0; i < history_length; i++) {
    // update time and reset trigger
    fake_time += time_delta;
    trigger = false;

    // invoke posterior handler
    ret = handler->Handle(new_posteriors, fake_time, &top_category_index, &trigger);

    // check if successful
    TEST_ASSERT_EQUAL(ESP_OK, ret);

    // comment in for debugging
    // printf("i=%u top=%u trigger=%x count=%u\n", i, top_category_index, trigger, trigger_count);

    // handle outputs
    if (trigger) {
      trigger_count++;
    }

    // after 24+-1 interations the label should schange from unknown to yes
    if (i < 23) {
      TEST_ASSERT_EQUAL(1, top_category_index);  // unknown
    } else if (i >= 23 && i <= 25) {
      // tol
    } else if (i > 25) {
      TEST_ASSERT_EQUAL(2, top_category_index);  // yes
    } else {
      // should never be reached
    }
  }

  // further assertions (suppression_ms=0 -> continuous triggering)
  TEST_ASSERT_EQUAL(history_length, trigger_count);
  TEST_ASSERT_EQUAL(2, top_category_index);  // yes

  // cleanup
  delete handler;
}

TEST_CASE("History length: 35, Threshold: 100, Suppression: 100 ms", "[default]") {
  // constants
  constexpr uint32_t history_length = 35;
  constexpr uint8_t trigger_threshold_single = 100;
  constexpr uint32_t suppression_ms = 100;
  constexpr uint32_t category_count = 4;

  // local working variables
  esp_err_t ret = ESP_OK;
  bool trigger = false;
  size_t trigger_count = 0;
  size_t top_category_index = 0;
  uint8_t new_posteriors[4] = {0};

  // dummy time for deterministic results
  constexpr uint32_t time_delta = 10;
  uint32_t fake_time = 0;

  // create instance of posterior handler to be tested
  PosteriorHandler *handler = new PosteriorHandler(history_length, trigger_threshold_single,
                                                   suppression_ms, category_count);

  // Fill with Silence (first 35 iterations have undefined behavior)
  new_posteriors[0] = 255;  // silence
  new_posteriors[1] = 0;    // unknown
  new_posteriors[2] = 0;    // yes
  new_posteriors[3] = 0;    // no
  trigger_count = 0;
  for (size_t i = 0; i < history_length; i++) {
    // update time and reset trigger
    trigger = false;
    fake_time += time_delta;

    // invoke posterior handler
    ret = handler->Handle(new_posteriors, fake_time, &top_category_index, &trigger);

    // check if successful
    TEST_ASSERT_EQUAL(ESP_OK, ret);

    // comment in for debugging
    // printf("i=%u top=%u trigger=%x count=%u\n", i, top_category_index, trigger, trigger_count);

    // handle outputs
    if (trigger) {
      trigger_count++;
    }
  }
  TEST_ASSERT_EQUAL(0, top_category_index);
  TEST_ASSERT_TRUE(trigger_count > 0);

  // Fill with Unknown
  new_posteriors[0] = 0;    // silence
  new_posteriors[1] = 255;  // unknown
  new_posteriors[2] = 0;    // yes
  new_posteriors[3] = 0;    // no
  trigger_count = 0;
  for (size_t i = 0; i < history_length; i++) {
    // update time and reset trigger
    trigger = false;
    fake_time += time_delta;

    // invoke posterior handler
    ret = handler->Handle(new_posteriors, fake_time, &top_category_index, &trigger);

    // check if successful
    TEST_ASSERT_EQUAL(ESP_OK, ret);

    // handle outputs
    if (trigger) {
      trigger_count++;
    }
    // comment in for debugging
    // printf("i=%u top=%u trigger=%x count=%u\n", i, top_category_index, trigger, trigger_count);

    // after 17+-1 interations the label should schange from silence to unknown
    if (i < 16) {
      TEST_ASSERT_EQUAL(0, top_category_index);  // yes
    } else if (i >= 16 && i <= 18) {
      // tol
    } else if (i > 18) {
      TEST_ASSERT_EQUAL(1, top_category_index);  // unknown
    } else {
      // should never be reached
    }
  }
  // further assertions (suppression_ms>0 -> expect between 1 and 3 triggers over complete window)
  TEST_ASSERT_UINT32_WITHIN(1, 3, trigger_count);
  TEST_ASSERT_EQUAL(1, top_category_index);  // unknown

  // Fill with Mixed
  new_posteriors[0] = 16;   // silence
  new_posteriors[1] = 32;   // unknown
  new_posteriors[2] = 144;  // yes
  new_posteriors[3] = 64;   // no
  trigger_count = 0;
  for (size_t i = 0; i < history_length; i++) {
    // update time and reset trigger
    trigger = false;
    fake_time += time_delta;

    // invoke posterior handler
    ret = handler->Handle(new_posteriors, fake_time, &top_category_index, &trigger);

    // check if successful
    TEST_ASSERT_EQUAL(ESP_OK, ret);

    // comment in for debugging
    // printf("i=%u top=%u trigger=%x count=%u\n", i, top_category_index, trigger, trigger_count);

    // handle outputs
    if (trigger) {
      trigger_count++;
    }

    // after 14+-1 interations the label should schange from silence to unknown
    if (i < 23) {
      TEST_ASSERT_EQUAL(1, top_category_index);  // unknown
    } else if (i >= 23 && i <= 25) {
      // tol
    } else if (i > 25) {
      TEST_ASSERT_EQUAL(2, top_category_index);  // yes
    } else {
      // should never be reached
    }
  }

  // further assertions (suppression_ms>0 -> expect between 1 and 4 triggers over complete window)
  TEST_ASSERT_UINT32_WITHIN(1, 4, trigger_count);
  TEST_ASSERT_EQUAL(2, top_category_index);  // yes

  // cleanup
  delete handler;
}
