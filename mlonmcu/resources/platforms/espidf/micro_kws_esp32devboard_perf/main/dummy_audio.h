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

#ifndef DUMMY_AUDIO_H
#define DUMMY_AUDIO_H

#include "esp_err.h"

bool DummyAudioRemaining(void);
esp_err_t GetDummyAudioData(size_t requested_size, size_t* actual_size, int8_t* data);
esp_err_t GetDummyAudioData_wrap(size_t requested_size, size_t* actual_size, int8_t* data);

#endif  // DUMMY_AUDIO_H