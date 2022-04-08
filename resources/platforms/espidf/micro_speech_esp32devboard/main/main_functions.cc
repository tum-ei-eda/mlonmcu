/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "main_functions.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "audio_provider.h"
#include "command_responder.h"
#include "feature_provider.h"
#include "micro_model_settings.h"
#include "recognize_commands.h"
#include "ml_interface.h"

// #include "tvm_wrapper.h"
// #include "tensorflow/lite/micro/micro_error_reporter.h"
// #include "tensorflow/lite/micro/micro_interpreter.h"
// #include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
// #include "tensorflow/lite/micro/system_setup.h"
// #include "tensorflow/lite/schema/schema_generated.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
// tflite::ErrorReporter* error_reporter = nullptr;
// const tflite::Model* model = nullptr;
// tflite::MicroInterpreter* interpreter = nullptr;
// TfLiteTensor* model_input = nullptr;
FeatureProvider* feature_provider = nullptr;
RecognizeCommands* recognizer = nullptr;
int32_t previous_time = 0;

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
// constexpr int kTensorArenaSize = 30 * 1024;
// uint8_t tensor_arena[kTensorArenaSize];
int8_t feature_buffer[kFeatureElementCount];
int8_t* model_input_buffer = nullptr;
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  mlif_init();
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  // static tflite::MicroErrorReporter micro_error_reporter;
  // error_reporter = &micro_error_reporter;

  // // Map the model into a usable data structure. This doesn't involve any
  // // copying or parsing, it's a very lightweight operation.
  // model = tflite::GetModel(g_model);
  // if (model->version() != TFLITE_SCHEMA_VERSION) {
  //   TF_LITE_REPORT_ERROR(error_reporter,
  //                        "Model provided is schema version %d not equal "
  //                        "to supported version %d.",
  //                        model->version(), TFLITE_SCHEMA_VERSION);
  //   return;
  // }

  // // Pull in only the operation implementations we need.
  // // This relies on a complete list of all the ops needed by this graph.
  // // An easier approach is to just use the AllOpsResolver, but this will
  // // incur some penalty in code space for op implementations that are not
  // // needed by this graph.
  // //
  // // tflite::AllOpsResolver resolver;
  // // NOLINTNEXTLINE(runtime-global-variables)
  // static tflite::MicroMutableOpResolver<4> micro_op_resolver(error_reporter);
  // if (micro_op_resolver.AddDepthwiseConv2D() != kTfLiteOk) {
  //   return;
  // }
  // if (micro_op_resolver.AddFullyConnected() != kTfLiteOk) {
  //   return;
  // }
  // if (micro_op_resolver.AddSoftmax() != kTfLiteOk) {
  //   return;
  // }
  // if (micro_op_resolver.AddReshape() != kTfLiteOk) {
  //   return;
  // }

  // // Build an interpreter to run the model with.
  // static tflite::MicroInterpreter static_interpreter(
  //     model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  // interpreter = &static_interpreter;

  // // Allocate memory from the tensor_arena for the model's tensors.
  // TfLiteStatus allocate_status = interpreter->AllocateTensors();
  // if (allocate_status != kTfLiteOk) {
  //   TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
  //   return;
  // }

  // // Get information about the memory area to use for the model's input.
  // model_input = interpreter->input(0);
  // if ((model_input->dims->size != 2) || (model_input->dims->data[0] != 1) ||
  //     (model_input->dims->data[1] !=
  //      (kFeatureSliceCount * kFeatureSliceSize)) ||
  //     (model_input->type != kTfLiteInt8)) {
  //   TF_LITE_REPORT_ERROR(error_reporter,
  //                        "Bad input tensor parameters in model");
  //   return;
  // }
  // model_input_buffer = model_input->data.int8;
  model_input_buffer = (int8_t*)mlif_input_ptr(0);

  // Prepare to access the audio spectrograms from a microphone or other source
  // that will provide the inputs to the neural network.
  // NOLINTNEXTLINE(runtime-global-variables)
  static FeatureProvider static_feature_provider(kFeatureElementCount,
                                                 feature_buffer);
  feature_provider = &static_feature_provider;

  // static RecognizeCommands static_recognizer(error_reporter);
  static RecognizeCommands static_recognizer;
  recognizer = &static_recognizer;

  previous_time = 0;
}

// The name of this function is important for Arduino compatibility.
void loop() {
  // Fetch the spectrogram for the current time.
  uint32_t start1 = xTaskGetTickCount();
  const int32_t current_time = LatestAudioTimestamp();
  int how_many_new_slices = 0;
  // TfLiteStatus feature_status = feature_provider->PopulateFeatureData(
  //     error_reporter, previous_time, current_time, &how_many_new_slices);
  int feature_status = feature_provider->PopulateFeatureData(
      previous_time, current_time, &how_many_new_slices);
  // printf("how_many_new_slices=%d\n", how_many_new_slices);
  if (feature_status != 0) {
    // TF_LITE_REPORT_ERROR(error_reporter, "Feature generation failed");
    printf("Feature generation failed\n");
    return;
  }
  // printf("current_time=%ld\n", current_time);
  previous_time = current_time;
  // If no new audio samples have been received since last time, don't bother
  // running the network model.
  if (how_many_new_slices == 0) {
    return;
  }
  uint32_t end1 = xTaskGetTickCount();
  // printf("ticks1=%lu\n", end1-start1);

  // Copy feature buffer to input tensor
  for (int i = 0; i < kFeatureElementCount; i++) {
    model_input_buffer[i] = feature_buffer[i];
  }

  // Run the model on the spectrogram input and make sure it succeeds.
  // TfLiteStatus invoke_status = interpreter->Invoke();
  // if (invoke_status != kTfLiteOk) {
  //   TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
  //   return;
  // }
  uint32_t start2 = xTaskGetTickCount();
  mlif_invoke();
  uint32_t end2 = xTaskGetTickCount();
  // printf("ticks2=%lu\n", end2-start2);

  // Obtain a pointer to the output tensor
  // TfLiteTensor* output = interpreter->output(0);
  int8_t* output_ptr = (int8_t*)mlif_output_ptr(0);
  printf("%4d, \t%d, \t%4d, \t%4d\n", output_ptr[0]+128, output_ptr[1]+128, output_ptr[2]+128, output_ptr[3]+128);
  // Determine whether a command was recognized based on the output of inference
  const char* found_command = nullptr;
  uint8_t score = 0;
  bool is_new_command = false;
  uint32_t start3 = xTaskGetTickCount();
  // TfLiteStatus process_status = recognizer->ProcessLatestResults(
  //     output, current_time, &found_command, &score, &is_new_command);
  int process_status = recognizer->ProcessLatestResults(
      output_ptr, current_time, &found_command, &score, &is_new_command);
  if (process_status != 0) {
    // TF_LITE_REPORT_ERROR(error_reporter,
    //                      "RecognizeCommands::ProcessLatestResults() failed");
    printf("RecognizeCommands::ProcessLatestResults() failed\n");
    return;
  }
  // Do something based on the recognized command. The default implementation
  // just prints to the error console, but you should replace this with your
  // own function for a real application.
  // RespondToCommand(error_reporter, current_time, found_command, score,
  //                  is_new_command);
  RespondToCommand(current_time, found_command, score,
                   is_new_command);
  uint32_t end3 = xTaskGetTickCount();
  // printf("ticks3=%lu\n", end3-start3);
}
