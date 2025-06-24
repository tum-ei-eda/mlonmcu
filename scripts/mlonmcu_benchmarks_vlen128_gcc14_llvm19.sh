#!/bin/bash
# TODO: docs and license

set -e

if [ "$#" -gt 2 ]; then
    echo "Illegal number of parameters"
    echo "Usage: $0 TARGET [MODEL1,MODEL2]"
    exit 1
fi
TARGET=${1:-"spike_rv64"}
MODELS=${2:-"aww,vww,resnet,toycar"}

SCRIPT_DIR="$( dirname -- "${BASH_SOURCE[0]}"; )"

TIMESTAMP=$(date +%Y%m%dT%H%M%S)

if [[ "$TARGET" == *"ssh"* ]]
then
    PARALLEL=1

else
    PARALLEL=$(($(nproc) / 4))
fi

for MODEL in ${MODELS//,/ }
do
    echo python $SCRIPT_DIR/gen_muriscnn_benchmarks.py $MODEL -b tflmi -t $TARGET --vlen 128 --post -p --parallel $PARALLEL -f muriscvnn -f vext -f auto_vectorize --toolchain gcc --toolchain llvm --label ${TIMESTAMP}-${TARGET}-${MODEL}-tflm-gccllvm --opt 3 --opt s --unroll 0 --unroll 1 --custom-unroll 0 --custom-unroll 1 --baseline 0 --skip-default
    python $SCRIPT_DIR/gen_muriscnn_benchmarks.py $MODEL -b tflmi -t $TARGET --vlen 128 --post -p --parallel $PARALLEL -f muriscvnn -f vext -f auto_vectorize --toolchain gcc --toolchain llvm --label ${TIMESTAMP}-${TARGET}-${MODEL}-tflm-gccllvm --opt 3 --opt s --unroll 0 --unroll 1 --custom-unroll 0 --custom-unroll 1 --baseline 0 --skip-default
done
