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
#include "printing.h"

#include <string.h>
#include <stdlib.h>

__attribute__((weak)) bool mlif_request_input(void *model_input_ptr, size_t model_input_sz) {
  static int num_done = 0;
  if (num_done == num_data_buffers_in) {
    static bool run_once = true;
    if (num_data_buffers_in == 0 && run_once) {
      // Minimal run. Just run the model without data.
      run_once = false;
      return true;
    }
    return false;
  }

  mlif_process_input(data_buffers_in[num_done], data_size_in[num_done], model_input_ptr, model_input_sz);
  num_done++;
  return true;
}

__attribute__((weak)) void mlif_handle_result(void *model_output_ptr, size_t model_output_sz) {
  static int num_done = 0;

  if (num_data_buffers_out == 0) {
    return;
  }

  if (num_done < num_data_buffers_out) {
    mlif_process_output(model_output_ptr, model_output_sz, data_buffers_out[num_done], data_size_out[num_done]);
  }

  num_done++;
}
