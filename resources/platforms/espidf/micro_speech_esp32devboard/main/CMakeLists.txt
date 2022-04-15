#[[
Copyright (c) 2022 TUM Department of Electrical and Computer Engineering.

This file is part of MLonMCU.
See https://github.com/tum-ei-eda/mlonmcu.git for further info.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
]]

# SET(COMPONENTS TODO)

# TODO: make chip/pins variable

idf_component_register(
  SRCS
  main.cc
  main_functions.cc
  audio_provider.cc
  feature_provider.cc
  recognize_commands.cc
  command_responder.cc
  micro_features_generator.cc
  micro_model_settings.cc
  ringbuf.c
  pins.c
  # TODO: use components?
  microfrontend/lib/fft.cc
  microfrontend/lib/fft_util.cc
  microfrontend/lib/filterbank.c
  microfrontend/lib/filterbank_util.c
  microfrontend/lib/frontend.c
  microfrontend/lib/frontend_util.c
  microfrontend/lib/kiss_fft_int16.cc
  microfrontend/lib/log_lut.c
  microfrontend/lib/log_scale.c
  microfrontend/lib/log_scale_util.c
  microfrontend/lib/noise_reduction.c
  microfrontend/lib/noise_reduction_util.c
  microfrontend/lib/pcan_gain_control_util.c
  microfrontend/lib/pcan_gain_control.c
  microfrontend/lib/window.c
  microfrontend/lib/window_util.c
  INCLUDE_DIRS
  .
  kissfft/
  REQUIRES
  mlif2
  spi_flash)
