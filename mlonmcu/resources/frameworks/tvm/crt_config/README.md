# Custom CRT Config

This `crt_config.h` can be used by the uTVMCG and the Target SW as a replacement to the default one (`tvm/apps/bundle_deploy/crt_config/crt_config.h`)

## Patches

- Increased `TVM_CRT_MAX_ARGS` from `10` to `30` to be able to use graph runtime together with TFLite Fallback
