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

#ifndef MODEL_SETTINGS_H
#define MODEL_SETTINGS_H

#include <cstdint>

#include "sdkconfig.h"

// The size of the input time series data we pass to the FFT to produce the
// frequency information. This has to be a power of two, and since we're dealing
// with 30ms of 16KHz inputs, which means 480 samples, this is the next larger
// value, i.e. 512.
constexpr int32_t max_audio_sample_size = 512;
constexpr int32_t audio_sample_frequency = 16000;

// The feature (powerspectrum image) on which the convolutional neural network
// operates on has 49 slices, each containing 40 grayscale pixels. So basically
// a 49 by 40 grayscale picture.
constexpr int32_t feature_slice_size = CONFIG_MICRO_KWS_NUM_BINS;
constexpr int32_t feature_slize_count = CONFIG_MICRO_KWS_NUM_SLICES;
constexpr int32_t feature_element_count = feature_slice_size * feature_slize_count;

constexpr int32_t feature_slice_stride_ms = CONFIG_MICRO_KWS_STRIDE_SIZE_MS;
constexpr int32_t feature_slice_duration_ms = CONFIG_MICRO_KWS_WINDOW_SIZE_MS;

constexpr int32_t category_count = CONFIG_MICRO_KWS_NUM_CLASSES;
extern const char* category_labels[category_count];

#endif  // MODEL_SETTINGS_H
