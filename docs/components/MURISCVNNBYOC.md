# muRISCV-NN BYOC Feature

## Usage

- The used extensions have to be manually selected by setting `muriscvnnbyoc.mcpu` on the command line
- Example mapping: `cortex-m55 -> VEXT`, `cortex-m33 -> PEXT`, `cortex-m0 -> No extensions`

## Compatibility

- The `muriscvnnbyoc` feature is not compatible with the `desired_layout` config for the TVM targets
