# Auto-Vectorize Feature

## Usage

- Add `-f auto_vectorize` to the command line arguments

## TODOs:

- [ ] Configure loop and basic block vectorizer individually.

## Warning
- Auto vectorization is enables by default (if available) on the following optimization levels:

  - GCC: `-O2`
  - LLVM: `-O1`

## Configuration

- `auto_vectorize.enable`: Allows to turn off ne auto-vectorization completely (Default: `true`)
- `auto_vectorize.verbose`: Print details about auto vectorization possibilities during compililation. Need to check the MLID stdout artifact or enable `mlif.print_outputs` to be effective (Default: `false`)

## Compatibility

- Only RISC-V targets is supported at the momemt
- The supported MLIF toolchains are GCC and LLVM
- A VLEN larger equals 128 is required for this feature
- It seems like this currently needs a ELEN=64 and proper alignment (e.g. `tvmaot.alignment_bytes=8`) for the backend data. TFLMI seems to break with this.
