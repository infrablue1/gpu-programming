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

# Use default M, N, K (4096, 4096, 4096)
for name in "${pass_names[@]}"
do
    $script --pass $name
    echo "---------------------------------------"
    sleep 1
done