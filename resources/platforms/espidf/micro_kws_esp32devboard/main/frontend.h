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

#ifndef FRONTEND_H
#define FRONTEND_H

#include <cstdint>

#include "esp_err.h"

// Sets up any resources needed for the feature generation pipeline.
esp_err_t InitializeFrontend();

// Converts audio sample data into a more compact form that's appropriate for
// feeding into a neural network.
esp_err_t GenerateFrontendData(const int16_t* input, size_t input_size, int8_t* output);

#endif  // FRONTEND_H
