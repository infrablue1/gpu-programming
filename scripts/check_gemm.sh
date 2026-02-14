#!/bin/bash
pass_names=("naive"
            "coalesced"
            "shared-memory"
            "blocktiling-1d"
            "blocktiling-2d"
            "vectorize"
            "warptiling"
            "double-buffer"
            "naive-tensorcore")

script="./build/matrix_mul/bench_gemm"
M=256
N=256
K=256

for name in "${pass_names[@]}"
do
    $script -M $M -N $N -K $K --pass $name --check-result
done