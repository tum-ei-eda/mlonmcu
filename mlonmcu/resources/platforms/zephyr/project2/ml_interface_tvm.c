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
