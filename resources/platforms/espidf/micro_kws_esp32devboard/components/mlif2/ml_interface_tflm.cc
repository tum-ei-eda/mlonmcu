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

// TODO: inline
// TODO: ideally we would just use #define aliases for those functions however tflm uses c++ linkage...
// #define mlif_input_ptr model_input_ptr
// #define mlif_input_size model_input_size
// #define mlif_inputs model_inputs
// #define mlif_invoke model_invoke
// #define mlif_output_ptr model_output_ptr
// #define mlif_output_size model_output_size
// #define mlif_outputs model_outputs

void mlif_init() {
  model_init();
}

void *mlif_input_ptr(int index) {
  return model_input_ptr(index);

}

size_t mlif_input_size(int index) {
  return model_input_size(index);
}

size_t mlif_inputs() {
  return model_inputs();
}

void mlif_invoke() {
  model_invoke();
}

void *mlif_output_ptr(int index) {
  return model_output_ptr(index);
}

size_t mlif_output_size(int index) {
  return model_output_size(index);
}

size_t mlif_outputs() {
  return model_outputs();
}
