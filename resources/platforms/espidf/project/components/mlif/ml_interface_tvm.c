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
#include "tvm_wrapper.h"

void mlif_run() {
  TVMWrap_Init();

  size_t input_num = 0;
  size_t remaining = NUM_RUNS;
  // inout data will only be applied in the first run!
  while (mlif_request_input(TVMWrap_GetInputPtr(input_num), TVMWrap_GetInputSize(input_num)) || remaining) {
    if (input_num == TVMWrap_GetNumInputs() - 1) {
      TVMWrap_Run();
      for (size_t i = 0; i < TVMWrap_GetNumOutputs(); i++) {
        mlif_handle_result(TVMWrap_GetOutputPtr(i), TVMWrap_GetOutputSize(i));
      }
      input_num = 0;
      remaining = remaining > 0 ? remaining - 1 : 0;
    } else {
      input_num++;
    }
  }
}
