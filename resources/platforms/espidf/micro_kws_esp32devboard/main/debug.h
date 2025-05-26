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

#ifndef DEBUG_H
#define DEBUG_H

#include <cstdint>

#include "esp_err.h"

// These parameters are relevant for the MICRO_KWS_MICROPHONE_DEBUG_MODE...

// If you incease this you probably also need to increase the micro_audio stack
// stack size.
#define AUDIO_SAMPLE_MS 2000
// 16bit audio @ 16kHz sample rate.
#define AUDIO_SAMPLE_SIZE (2 * 16 * AUDIO_SAMPLE_MS)
// 16bit audio @ 16kHz sample rate.
#define AUDIO_PACKET_SIZE (2 * 16 * 100)  // Sending 100ms at once to host PC

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

esp_err_t DebugRun(int8_t* feature_data, uint8_t* category_data, uint8_t top_category_index);

esp_err_t InitializeDebug();

esp_err_t StopDebug();

esp_err_t DebugRunAudio(int8_t* audio_data);

void micro_audio(void* params);

#endif  // DEBUG_H
