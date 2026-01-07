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

#ifdef __cplusplus
extern "C"
#endif
int mlonmcu_init() {
  return model_init();
}

#ifdef __cplusplus
extern "C"
#endif
int mlonmcu_deinit() {
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
int mlonmcu_run() {
  size_t remaining = NUM_RUNS;
  int ret;
  while (remaining) {
    ret = model_invoke();
    if (ret) {
      return ret;
    }
    remaining--;
  }
  return ret;
}

#ifdef __cplusplus
extern "C"
#endif
int mlonmcu_check() {
  size_t input_num = 0;
  int ret;
  bool new_;
  while (true) {
    ret = mlif_request_input(model_input_ptr(input_num), model_input_size(input_num), &new_);
    if (ret) {
      return ret;
    }
    if (!new_) {
      break;
    }
    if (input_num == model_inputs() - 1) {
      ret = model_invoke();
      if (ret) {
        return ret;
      }
      for (size_t i = 0; i < model_outputs(); i++) {
        ret = mlif_handle_result(model_output_ptr(i), model_output_size(i));
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
