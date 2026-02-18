# CUDA Profiling & Debugging

## Profiling Tools Overview

| Tool             | Purpose                                     | When to Use                           |
|-----------------|---------------------------------------------|---------------------------------------|
| Nsight Systems   | System-wide timeline profiling               | Start here: identify CPU/GPU overlap  |
| Nsight Compute   | Kernel-level performance analysis            | Deep-dive into specific kernel perf   |
| nvprof (legacy)  | Command-line profiling                       | Quick checks, CI pipelines            |
| cuda-memcheck    | Memory error detection                       | Debugging OOB, races, leaks           |
| compute-sanitizer| Modern replacement for cuda-memcheck         | Same, CUDA 11.6+                      |

## Nsight Systems

System-wide profiler for CPU-GPU interaction, stream utilization, and kernel timeline:

```bash
# Profile an application
nsys profile -o report ./my_cuda_app

# With CUDA API tracing
nsys profile --trace=cuda,nvtx,osrt -o report ./my_cuda_app

# Generate summary stats
nsys stats report.nsys-rep
```

### NVTX Annotations

Mark regions in your code for visibility in the profiler timeline:

```cuda
#include <nvtx3/nvToolsExt.h>

void trainEpoch(Model& model) {
    nvtxRangePush("Forward Pass");
    forward(model);
    nvtxRangePop();

    nvtxRangePush("Backward Pass");
    backward(model);
    nvtxRangePop();

    nvtxRangePush("Weight Update");
    update(model);
    nvtxRangePop();
}
```

```bash
# Link with NVTX
nvcc -lnvToolsExt -o app app.cu
```

## Nsight Compute

Detailed per-kernel metrics: occupancy, memory throughput, compute utilization, warp stall reasons.

```bash
# Profile all kernels
ncu ./my_cuda_app

# Profile specific kernel with full metrics
ncu --set full --kernel-name myKernel ./my_cuda_app

# Compare two runs
ncu --import baseline.ncu-rep --import optimized.ncu-rep

# Export to file
ncu -o report --set full ./my_cuda_app
```

### Key Metrics to Check

- **Achieved Occupancy** -- ratio of active warps to max possible
- **Memory Throughput** -- % of theoretical peak bandwidth
- **Compute Throughput** -- % of theoretical peak FLOPS
- **Warp Stall Reasons** -- why warps are waiting (memory, sync, instruction fetch)
- **L1/L2 Cache Hit Rates** -- indicates memory access pattern efficiency
- **Shared Memory Bank Conflicts** -- serialization due to conflicts
- **Register Spills** -- registers spilling to local memory (slow)

### Roofline Analysis

Nsight Compute provides roofline charts. Key concepts:

- **Arithmetic Intensity** = FLOP / bytes transferred
- **Memory-bound** kernel: intensity below ridge point -- optimize memory access
- **Compute-bound** kernel: intensity above ridge point -- optimize instructions

## compute-sanitizer

Modern memory checking tool (replaces cuda-memcheck):

```bash
# Check for memory errors (out-of-bounds, misaligned access)
compute-sanitizer --tool memcheck ./my_cuda_app

# Race condition detection
compute-sanitizer --tool racecheck ./my_cuda_app

# Shared memory hazard detection
compute-sanitizer --tool synccheck ./my_cuda_app

# Memory leak detection
compute-sanitizer --tool initcheck ./my_cuda_app
```

## Error Handling Patterns

### Comprehensive Error Checking Macro

```cuda
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in %s at line %d: %s (%d)\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err), err);          \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// For kernel launches (asynchronous errors)
#define CUDA_CHECK_KERNEL()                                                    \
    do {                                                                       \
        CUDA_CHECK(cudaGetLastError());                                        \
        CUDA_CHECK(cudaDeviceSynchronize());                                   \
    } while (0)
```

### Debug vs Release Error Checking

```cuda
#ifdef NDEBUG
#define CUDA_DEBUG_CHECK(call) (call)
#else
#define CUDA_DEBUG_CHECK(call) CUDA_CHECK(call)
#endif

// Expensive sync-based check only in debug builds
#ifdef NDEBUG
#define CUDA_DEBUG_SYNC()
#else
#define CUDA_DEBUG_SYNC() CUDA_CHECK(cudaDeviceSynchronize())
#endif
```

### Checking Device Properties

```cuda
void checkDevice(int deviceId) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));

    printf("Device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("SMs: %d\n", prop.multiProcessorCount);
    printf("Max threads/block: %d\n", prop.maxThreadsPerBlock);
    printf("Max threads/SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Shared memory/block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("Shared memory/SM: %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("Registers/block: %d\n", prop.regsPerBlock);
    printf("Warp size: %d\n", prop.warpSize);
    printf("Global memory: %.1f GB\n", prop.totalGlobalMem / 1e9);
    printf("Memory bus width: %d bits\n", prop.memoryBusWidth);
    printf("L2 cache size: %d KB\n", prop.l2CacheSize / 1024);
    printf("Clock rate: %.1f GHz\n", prop.clockRate / 1e6);

    // Check peer access
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for (int peer = 0; peer < deviceCount; peer++) {
        if (peer != deviceId) {
            int canAccess;
            cudaDeviceCanAccessPeer(&canAccess, deviceId, peer);
            printf("Peer access to device %d: %s\n", peer, canAccess ? "yes" : "no");
        }
    }
}
```

## Common Debugging Techniques

### Printf from Kernels

```cuda
__global__ void debugKernel(float* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && i < 10) {  // limit output!
        printf("Thread %d: data[%d] = %f\n", i, i, data[i]);
    }
}
// Requires compute capability 2.0+
// Output appears after next cudaDeviceSynchronize() or kernel completion
```

### Validating Results

```cuda
void validate(float* h_gpu, float* h_cpu, int N, float tolerance = 1e-5f) {
    int errors = 0;
    float maxDiff = 0;
    for (int i = 0; i < N; i++) {
        float diff = fabsf(h_gpu[i] - h_cpu[i]);
        if (diff > tolerance) {
            if (errors < 10)  // print first 10 errors
                printf("Mismatch at %d: GPU=%.8f CPU=%.8f diff=%.8e\n",
                       i, h_gpu[i], h_cpu[i], diff);
            errors++;
        }
        maxDiff = fmaxf(maxDiff, diff);
    }
    printf("%d errors out of %d (max diff: %.8e)\n", errors, N, maxDiff);
}
```

## Compilation Flags

```bash
# Debug build (device debug info, host debug)
nvcc -G -g -O0 -lineinfo -o debug_app kernel.cu

# Release build
nvcc -O3 -use_fast_math -o release_app kernel.cu

# Show resource usage (registers, shared memory per kernel)
nvcc --resource-usage -o app kernel.cu

# Generate PTX for inspection
nvcc --ptx -o kernel.ptx kernel.cu

# Target specific architecture
nvcc -arch=sm_89 -o app kernel.cu     # Ada Lovelace (RTX 4090)
nvcc -arch=sm_90 -o app kernel.cu     # Hopper (H100)
nvcc -arch=sm_100 -o app kernel.cu    # Blackwell (B200)

# JIT compilation for forward compatibility
nvcc -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_89,code=sm_89 \
     -gencode arch=compute_90,code=compute_90 \
     -o app kernel.cu
```

## Performance Checklist

Before considering a CUDA kernel optimized, verify:

- [ ] **Coalesced memory access** -- no strided or random global reads/writes
- [ ] **Occupancy** -- at least 50% (use occupancy calculator)
- [ ] **No warp divergence** -- check with Nsight Compute stall analysis
- [ ] **Shared memory** -- used where data is reused, no bank conflicts
- [ ] **Minimal host-device transfers** -- batch data, use async copies
- [ ] **Error checking** -- all API calls checked in debug builds
- [ ] **Appropriate precision** -- float vs double justified
- [ ] **Stream overlap** -- compute and memory transfers overlap where possible
- [ ] **No register spills** -- check with `--resource-usage`
- [ ] **Kernel fusion** -- unnecessary intermediate buffers eliminated
