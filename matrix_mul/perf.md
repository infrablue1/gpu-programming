

For gemm shape M=N=K=4096:
|Kernel|GFLOPS|Kernel time(ms)|
|-|-|-|
|naive|139.34|986|
|coalesced|1036.76|132.5|
|shared_memory|1342.81|102.3|
|blocktiling-1d|2864.56|48.13|
|blocktiling-2d|4014.83|34.23|

