

For gemm shape M=N=K=4096:
|Kernel|GFLOPS|Kernel time(ms)|
|-|-|-|
|naive|139.34|986|
|coalesced|1036.76|132.5|
|shared_memory|1342.81|102.3|
|blocktiling-1d|3517.76|39.07|
|blocktiling-2d|7405.25|18.56|
|vectorize|7837.50|17.53|
