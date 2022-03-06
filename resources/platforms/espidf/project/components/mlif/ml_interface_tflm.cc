/*
 * Copyright (c) 2022 TUM Department of Electrical and Computer Engineering.
 *
 * This file is part of MLonMCU.
 * See https://github.com/tum-ei-eda/mlonmcu.git for further info.
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
#include "ml_interface.h"

#include "model.cc.h"

void mlif_run() {
  model_init();

  size_t input_num = 0;
  size_t remaining = NUM_RUNS;
  while (mlif_request_input(model_input_ptr(input_num), model_input_size(input_num)) || remaining) {
    if (input_num == model_inputs() - 1) {
      model_invoke();
      for (size_t i = 0; i < model_outputs(); i++) {
        mlif_handle_result(model_output_ptr(i), model_output_size(i));
      }
      input_num = 0;
      remaining = remaining > 0 ? remaining - 1 : 0;
    } else {
      input_num++;
    }
  }
}
