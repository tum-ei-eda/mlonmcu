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

int mlonmcu_init() {
  return TVMWrap_Init();
}
int mlonmcu_deinit() {return 0;}

int mlonmcu_run() {
  size_t remaining = NUM_RUNS;
  while (remaining) {
    int ret = TVMWrap_Run();
    if (ret) {
      return ret;
    }
    remaining--;
  }
  return 0;
}

int mlonmcu_check() {
  size_t input_num = 0;
  int ret = 0;
  bool new_;
  while (true) {
    ret = mlif_request_input(TVMWrap_GetInputPtr(input_num), TVMWrap_GetInputSize(input_num), &new_);
    if (ret) {
      return ret;
    }
    if (!new_) {
      break;
    }
    if (input_num == TVMWrap_GetNumInputs() - 1) {
      ret = TVMWrap_Run();
      if (ret) {
        return ret;
      }
      for (size_t i = 0; i < TVMWrap_GetNumOutputs(); i++) {
        ret = mlif_handle_result(TVMWrap_GetOutputPtr(i), TVMWrap_GetOutputSize(i));
        if (ret) {
          return ret;
        }
      }
      input_num = 0;
    } else {
      input_num++;
    }
  }
  return ret;
}
