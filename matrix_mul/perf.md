
Use scripts to check results or get performance:

```bash
# check result
./scripts/check_gemm.sh 

# get perf
./scripts/bench_gemm.sh 
```

For gemm shape M=N=K=4096:
|Kernel|GFLOPS|Kernel time(ms)|
|-|-|-|
|naive|139.34|986|
|coalesced|1036.76|132.5|
|shared_memory|1342.81|102.3|
|blocktiling-1d|3517.76|39.07|
|blocktiling-2d|7405.25|18.56|
|vectorize|9721.29|14.138|
|warptiling|10678.90|12.87|
