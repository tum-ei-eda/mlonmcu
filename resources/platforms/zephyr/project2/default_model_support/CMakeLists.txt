ADD_LIBRARY(model_support STATIC process_input.c process_output.c)
# TARGET_LINK_LIBRARIES(model_support PUBLIC ml_interface_base)
TARGET_INCLUDE_DIRECTORIES(model_support PUBLIC ../lib/support)
