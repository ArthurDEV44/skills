---
name: cuda-best-practices
description: >
  NVIDIA CUDA C/C++ GPU programming best practices and performance optimization. Covers kernel design,
  thread/block/grid hierarchy, memory hierarchy (global, shared, constant, texture, registers, L1/L2),
  coalescing, bank conflicts, occupancy, warp divergence, streams, CUDA graphs, cooperative groups,
  unified/pinned memory, error handling, Nsight profiling, atomics, dynamic parallelism, Thrust/CUB,
  and modern CUDA 12+ features. Use when writing, reviewing, or optimizing CUDA code: (1) Kernel
  launch configuration, (2) Memory access optimization, (3) Shared memory and bank conflicts,
  (4) Streams and async concurrency, (5) Profiling and debugging, (6) Error handling,
  (7) Unified vs pinned vs device memory, (8) Cooperative groups and CUDA graphs,
  (9) Warp-level primitives and control flow, (10) Thrust/CUB and cuBLAS usage.
---

# CUDA Best Practices

## Core Optimization Strategy

Performance optimization revolves around three pillars:

1. **Maximize parallel execution** -- expose parallelism, choose optimal launch config, use streams
2. **Optimize memory throughput** -- coalesce accesses, use shared memory, minimize host-device transfers
3. **Optimize instruction throughput** -- avoid divergence, use intrinsics, prefer single precision

Always **profile first** before optimizing. Use Nsight Compute and Nsight Systems to identify bottlenecks.

## Thread Hierarchy & Launch Configuration

### Grid, Block, Thread

```cuda
// 1D launch -- most common
int N = 1 << 20;
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
myKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);

// 2D launch -- for matrices, images
dim3 block(16, 16);          // 256 threads per block
dim3 grid((width + 15) / 16, (height + 15) / 16);
matrixKernel<<<grid, block>>>(d_out, d_in, width, height);

// 3D launch -- for volumes
dim3 block3(8, 8, 8);        // 512 threads per block
dim3 grid3((X+7)/8, (Y+7)/8, (Z+7)/8);
volumeKernel<<<grid3, block3>>>(d_vol, X, Y, Z);
```

### Choosing Block Size

- Use multiples of 32 (warp size) -- **128, 256, or 512** are typical good choices
- Use the occupancy calculator to find the optimal value:

```cuda
int minGridSize, blockSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, myKernel, 0, 0);
int gridSize = (N + blockSize - 1) / blockSize;
myKernel<<<gridSize, blockSize>>>(args);
```

### Occupancy

Occupancy = active warps / max warps per SM. Higher occupancy helps hide memory latency.

```cuda
int numBlocksPerSm = 0;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, myKernel, blockSize, sharedMemBytes);
```

Occupancy limiters: registers per thread, shared memory per block, threads per block. Use `--resource-usage` flag with `nvcc` to inspect.

## Memory Hierarchy

### Global Memory -- Coalescing is Critical

Coalesced access: threads in a warp access consecutive addresses in the same 128-byte cache line.

```cuda
// GOOD: coalesced -- thread i accesses element i
__global__ void coalesced(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) data[idx] *= 2.0f;
}

// BAD: strided -- each thread skips stride elements
__global__ void strided(float* data, int N, int stride) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    if (idx < N) data[idx] *= 2.0f;  // wastes bandwidth
}
```

**Structure of Arrays (SoA) over Array of Structures (AoS):**

```cuda
// BAD: AoS -- non-coalesced field access
struct Particle { float x, y, z, w; };
Particle* particles;  // particles[i].x is strided

// GOOD: SoA -- coalesced per-field access
struct Particles {
    float* x; float* y; float* z; float* w;
};
// x[i] is contiguous
```

### Shared Memory

Fast on-chip memory shared within a block. Use for data reuse and as a manual cache.

```cuda
__global__ void transpose(float* out, const float* in, int width) {
    __shared__ float tile[32][33];  // +1 padding to avoid bank conflicts

    int xIdx = blockIdx.x * 32 + threadIdx.x;
    int yIdx = blockIdx.y * 32 + threadIdx.y;

    // Coalesced read from global -> shared
    tile[threadIdx.y][threadIdx.x] = in[yIdx * width + xIdx];
    __syncthreads();

    // Transposed coalesced write from shared -> global
    xIdx = blockIdx.y * 32 + threadIdx.x;
    yIdx = blockIdx.x * 32 + threadIdx.y;
    out[yIdx * width + xIdx] = tile[threadIdx.x][threadIdx.y];
}
```

**Bank conflicts:** Shared memory is divided into 32 banks. If multiple threads in a warp access different addresses in the same bank, accesses serialize. Pad arrays with +1 to avoid conflicts (e.g., `[32][33]` instead of `[32][32]`).

### Constant Memory

64KB read-only cache, broadcast to all threads in a warp. Use for coefficients, lookup tables:

```cuda
__constant__ float filter[256];

// Host side
cudaMemcpyToSymbol(filter, h_filter, 256 * sizeof(float));
```

### Registers

Fastest per-thread storage. Excessive register usage reduces occupancy (register spilling to local memory).

For detailed memory optimization patterns, bank conflict resolution, and unified memory guidance, see `references/memory-optimization.md`.

## Warp Execution & Control Flow

### Avoid Warp Divergence

All 32 threads in a warp execute the same instruction. Divergent branches serialize:

```cuda
// BAD: divergent -- half the warp idles
if (threadIdx.x % 2 == 0) { doA(); } else { doB(); }

// BETTER: diverge on warp boundaries
if (threadIdx.x / 32 < N/2) { doA(); } else { doB(); }

// BEST: restructure to avoid branching entirely
float result = condition ? valA : valB;  // predication, no divergence
```

### Warp-Level Primitives

```cuda
// Warp shuffle -- exchange data within a warp without shared memory
int val = __shfl_down_sync(0xFFFFFFFF, myVal, offset);

// Warp vote
int allTrue = __all_sync(0xFFFFFFFF, predicate);
int anyTrue = __any_sync(0xFFFFFFFF, predicate);
unsigned ballot = __ballot_sync(0xFFFFFFFF, predicate);

// Warp reduce (example: sum within a warp)
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}
```

## Streams & Async Concurrency

### Overlapping Compute and Transfers

```cuda
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// Overlap: copy chunk1 while processing chunk0
cudaMemcpyAsync(d_in0, h_in0, size, cudaMemcpyHostToDevice, stream1);
kernel<<<grid, block, 0, stream1>>>(d_in0, d_out0);
cudaMemcpyAsync(h_out0, d_out0, size, cudaMemcpyDeviceToHost, stream1);

cudaMemcpyAsync(d_in1, h_in1, size, cudaMemcpyHostToDevice, stream2);
kernel<<<grid, block, 0, stream2>>>(d_in1, d_out1);
cudaMemcpyAsync(h_out1, d_out1, size, cudaMemcpyDeviceToHost, stream2);

cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);
cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
```

**Pinned (page-locked) memory** is required for async transfers:

```cuda
float* h_data;
cudaMallocHost(&h_data, size);  // pinned allocation
// ... use with cudaMemcpyAsync ...
cudaFreeHost(h_data);
```

For streams, events, CUDA graphs, and advanced concurrency, see `references/async-concurrency.md`.

## Error Handling

### Always Check Errors

```cuda
// Macro for runtime API calls
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Usage
CUDA_CHECK(cudaMalloc(&d_data, size));
CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

// Kernel launch errors (asynchronous -- check after sync)
myKernel<<<grid, block>>>(args);
CUDA_CHECK(cudaGetLastError());        // launch config errors
CUDA_CHECK(cudaDeviceSynchronize());   // runtime errors
```

### Common Pitfalls

- **Forgetting bounds checks:** Always guard with `if (idx < N)` in kernels
- **Missing `__syncthreads()`:** Required before reading shared memory written by other threads
- **Race conditions:** Use atomics or synchronization when multiple threads write to the same location
- **Not checking kernel launch errors:** `cudaGetLastError()` after every launch
- **Host-device pointer confusion:** Never dereference device pointers on host or vice versa

## Common Anti-Patterns

### Excessive Host-Device Transfers

```cuda
// BAD: transfer for each element
for (int i = 0; i < N; i++) {
    cudaMemcpy(&d_data[i], &h_data[i], sizeof(float), cudaMemcpyHostToDevice);
}

// GOOD: single bulk transfer
cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
```

### Launching Too Few Threads

```cuda
// BAD: underutilizes GPU
kernel<<<1, 1>>>(data, N);  // single thread doing serial work

// GOOD: parallelize across many threads
kernel<<<(N+255)/256, 256>>>(data, N);
```

### Ignoring Alignment

```cuda
// BAD: unaligned access wastes bandwidth
struct __align__(4) Bad { char a; float b; };  // float misaligned

// GOOD: naturally aligned
struct __align__(8) Good { float b; char a; };  // float at offset 0
```

### Serial Bottlenecks

```cuda
// BAD: serial reduction on GPU
__global__ void badReduce(float* data, float* result, int N) {
    if (threadIdx.x == 0) {
        float sum = 0;
        for (int i = 0; i < N; i++) sum += data[i];
        *result = sum;
    }
}

// GOOD: parallel reduction (see references/kernel-optimization.md)
```

For kernel optimization patterns (reductions, tiling, loop unrolling, instruction throughput), see `references/kernel-optimization.md`.

For profiling and debugging guidance (Nsight, nvprof, cuda-memcheck), see `references/profiling-debugging.md`.

For modern CUDA features (cooperative groups, CUDA graphs, dynamic parallelism, Thrust/CUB), see `references/modern-features.md`.
