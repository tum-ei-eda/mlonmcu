#ifndef ML_INTERFACE_DEFS_H
#define ML_INTERFACE_DEFS_H

#include "tvm_wrapper.h"

#define mlif_init TVMWrap_Init
#define mlif_input_ptr TVMWrap_GetInputPtr
#define mlif_input_size TVMWrap_GetInputSize
#define mlif_inputs TVMWrap_GetNumInputs
#define mlif_invoke TVMWrap_Run
#define mlif_output_ptr TVMWrap_GetOutputPtr
#define mlif_output_size TVMWrap_GetOutputSize
#define mlif_outputs TVMWrap_GetNumOutputs

#endif
