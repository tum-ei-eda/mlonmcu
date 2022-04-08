#ifndef ML_INTERFACE_DEFS_H
#define ML_INTERFACE_DEFS_H

#include "model.cc.h"

#define mlif_init model_Init
#define mlif_input_ptr model_input_ptr
#define mlif_input_size model_input_size
#define mlif_inputs model_inputs
#define mlif_invoke model_invoke
#define mlif_output_ptr model_output_ptr
#define mlif_output_size model_output_size
#define mlif_outputs model_outputs

#endif
