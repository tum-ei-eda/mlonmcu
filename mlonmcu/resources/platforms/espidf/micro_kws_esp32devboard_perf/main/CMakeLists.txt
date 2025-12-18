#[[
Copyright(c) 2022 TUM Department of Electrical and Computer Engineering
    .

    This file is part of MLonMCU.See https
    :  // github.com/tum-ei-eda/mlonmcu.git for further info.

       Licensed under the Apache License,
    Version 2.0(the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
]]

set(MICROFRONTEND_DIR microfrontend)

set(MICROFRONTEND_SRCS
    ${MICROFRONTEND_DIR}/lib/fft.cc
    ${MICROFRONTEND_DIR}/lib/fft_util.cc
    ${MICROFRONTEND_DIR}/lib/filterbank.c
    ${MICROFRONTEND_DIR}/lib/filterbank_util.c
    ${MICROFRONTEND_DIR}/lib/frontend.c
    ${MICROFRONTEND_DIR}/lib/frontend_util.c
    ${MICROFRONTEND_DIR}/lib/kiss_fft_int16.cc
    ${MICROFRONTEND_DIR}/lib/log_lut.c
    ${MICROFRONTEND_DIR}/lib/log_scale.c
    ${MICROFRONTEND_DIR}/lib/log_scale_util.c
    ${MICROFRONTEND_DIR}/lib/noise_reduction.c
    ${MICROFRONTEND_DIR}/lib/noise_reduction_util.c
    ${MICROFRONTEND_DIR}/lib/pcan_gain_control_util.c
    ${MICROFRONTEND_DIR}/lib/pcan_gain_control.c
    ${MICROFRONTEND_DIR}/lib/window.c
    ${MICROFRONTEND_DIR}/lib/window_util.c
)

set(KISSFFT_INCS kissfft/ kissfft/tools/)

# set(MLF_DIR ${CMAKE_SOURCE_DIR}/model/)

# set(CRT_DIR ${CMAKE_SOURCE_DIR}/crt/)

# set(TVM_SRCS ${MLF_DIR}/codegen/host/src/default_lib0.c ${MLF_DIR}/codegen/host/src/default_lib1.c)

# set(TVM_INCS ${CRT_DIR}/include/)

# if (EXISTS ${MLF_DIR}/codegen/host/include)
    # set(TVM_INCS ${TVM_INCS} ${MLF_DIR}/codegen/host/include/)
# endif()

# set(MICRO_KWS_SRCS audio.cc backend.cc debug.cc frontend.cc gpio.cc model_settings.cc tvm_wrapper.cc)
set(MICRO_KWS_SRCS audio.cc backend.cc bench.cc debug.cc dummy_audio.cc frontend.cc gpio.cc model_settings.cc wifi_udp.cc)


idf_component_register(
    SRCS
    main.cc
    ${MICRO_KWS_SRCS}
    ${MICROFRONTEND_SRCS}
    # ${TVM_SRCS}
    INCLUDE_DIRS
    .
    ${KISSFFT_INCS}
    # ${TVM_INCS}
    # Workaround:
    ${CMAKE_CURRENT_SOURCE_DIR}/../components/mlif2/include/
    REQUIRES
    driver
    spi_flash
    posterior
    mlif2
    esp_wifi  # Add WiFi component
    esp_event # Add event loop component
    nvs_flash # Add NVS flash component for WiFi credentials storage
    esp_http_client
    wpa_supplicant  # Add this line to include the wpa_supplicant component
)
# get_target_property(MLIF_INCS mlif2 INCLUDE_DIRECTORIES)
# target_include_directories(${COMPONENT_LIB} PUBLIC ${MLIF_INCS})

target_compile_options(${COMPONENT_LIB} PRIVATE -Wno-incompatible-pointer-types)

set(PCER_INIT_VAL "1" CACHE STRING "Performance counter setting")
set(ENABLE_PERF_EVAL "0" CACHE STRING "Performance evaluation setting")
set(CONFIG_ENABLE_WIFI "0" CACHE STRING "Enable Wi-Fi functionality")
target_compile_definitions(${COMPONENT_TARGET} PRIVATE "-DPCER_INIT_VAL=${PCER_INIT_VAL}"  "-DENABLE_PERF_EVAL=${ENABLE_PERF_EVAL}"  "-DCONFIG_ENABLE_WIFI=${CONFIG_ENABLE_WIFI}")
