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

#include "frontend.h"

#include <cmath>
#include <cstdio>
#include <cstring>

#include "esp_log.h"
#include "microfrontend/lib/frontend.h"
#include "microfrontend/lib/frontend_util.h"
#include "model_settings.h"

FrontendState micro_features_state;

esp_err_t InitializeFrontend() {
  // TODO(fabianpedd): Understand each value and try to finetune it. Refer to
  // the paper: "TRAINABLE FRONTEND FOR ROBUST AND FAR-FIELD KEYWORD SPOTTING"
  // TODO(fabianpedd): Check if the training in Keras, where the frontend is
  // part of the model, is changing/optimizing these parameters during training
  FrontendConfig config;
  config.window.size_ms = feature_slice_duration_ms;
  config.window.step_size_ms = feature_slice_stride_ms;
  config.filterbank.num_channels = feature_slice_size;
  config.filterbank.lower_band_limit = 125.0;
  config.filterbank.upper_band_limit = 7500.0;
  config.noise_reduction.smoothing_bits = 10;
  config.noise_reduction.even_smoothing = 0.025;
  config.noise_reduction.odd_smoothing = 0.06;
  config.noise_reduction.min_signal_remaining = 0.05;
  config.pcan_gain_control.enable_pcan = 1;
  config.pcan_gain_control.strength = 0.95;
  config.pcan_gain_control.offset = 80.0;
  config.pcan_gain_control.gain_bits = 21;
  config.log_scale.enable_log = 1;
  config.log_scale.scale_shift = 6;

  if (!FrontendPopulateState(&config, &micro_features_state, audio_sample_frequency)) {
    ESP_LOGE(__FILE__, "ERROR: FrontendPopulateState() failed.");
    return ESP_FAIL;
  }
  return ESP_OK;
}

esp_err_t GenerateFrontendData(const int16_t* input, size_t input_size, int8_t* output) {
  // TODO(fabianpedd): Simply add the 160 directly to input without the need for
  // an extra variable. But was is this code here for anyways!?!
  const int16_t* frontend_input;
  static bool is_first_time = true;
  if (is_first_time) {
    frontend_input = input;
    is_first_time = false;
  } else {
    // TODO(fabianpedd): Why do we need to add 160 here? Has this smth todo with
    // the comment from petewarden in feature_provider?
    frontend_input = input + 160;
  }

  // TODO(fabianpedd): Unused...
  size_t num_samples_read = 0;
  (void)num_samples_read;

  FrontendOutput frontend_output =
      FrontendProcessSamples(&micro_features_state, frontend_input, input_size, &num_samples_read);

  if (frontend_output.size <= 0 || frontend_output.values == NULL) {
    ESP_LOGE(__FILE__, "ERROR: In FrontendProcessSamples().");
    return ESP_FAIL;
  }

  for (size_t i = 0; i < frontend_output.size; ++i) {
    // TODO(fabianpedd): Figure out what exactly is happening here with the
    // rounding and if this is related to the performance discrepancies between
    // training and inference.

    // These scaling values are derived from those used in
    // input_data.py in the training pipeline. The feature pipeline outputs
    // 16-bit signed integers in roughly a 0 to 670 range. In training, these
    // are then arbitrarily divided by 25.6 to get float values in the rough
    // range of 0.0 to 26.0. This scaling is performed for historical reasons,
    // to match up with the output of other feature generators. The process is
    // then further complicated when we quantize the model. This means we have
    // to scale the 0.0 to 26.0 real values to the -128 to 127 signed integer
    // numbers. All this means that to get matching values from our integer
    // feature output into the tensor input, we have to perform: input =
    // (((feature / 25.6) / 26.0) * 256) - 128 To simplify this and perform it
    // in 32-bit integer math, we rearrange to: input = (feature * 256) / (25.6
    // * 26.0) - 128
    constexpr int32_t value_scale = 256;
    constexpr int32_t value_div = static_cast<int32_t>((25.6f * 26.0f) + 0.5f);
    int32_t value = ((frontend_output.values[i] * value_scale) + (value_div / 2)) / value_div;
    value -= 128;
    if (value < -128) {
      value = -128;
    }
    if (value > 127) {
      value = 127;
    }
    output[i] = value;
  }

  return ESP_OK;
}
