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

#ifndef BACKEND_H
#define BACKEND_H

#include <cstdint>

#include "esp_err.h"
#include "model_settings.h"

/**
 * @brief
 *
 * @param new_posteriors The raw model outputs with unsigned 8-bit values.
 * @param top_category_index The index of the detected category/label returned by reference.
 *
 * @return ESP_OK if no error occured.
 *
 */
esp_err_t HandlePosteriors(uint8_t new_posteriors[category_count], size_t* top_category_index);

#endif  // BACKEND_H
