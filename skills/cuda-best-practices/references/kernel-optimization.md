# CUDA Kernel Optimization

## Parallel Reduction

The canonical pattern for summing/reducing an array on GPU:

### Tree-Based Reduction (Shared Memory)

```cuda
__global__ void reduce(float* input, float* output, int N) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Load two elements per thread (improves occupancy)
    float sum = 0.0f;
    if (i < N) sum = input[i];
    if (i + blockDim.x < N) sum += input[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    // Tree reduction in shared memory
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Final warp reduction (no __syncthreads needed within a warp)
    if (tid < 32) {
        float val = sdata[tid];
        val += __shfl_down_sync(0xFFFFFFFF, val, 16);
        val += __shfl_down_sync(0xFFFFFFFF, val, 8);
        val += __shfl_down_sync(0xFFFFFFFF, val, 4);
        val += __shfl_down_sync(0xFFFFFFFF, val, 2);
        val += __shfl_down_sync(0xFFFFFFFF, val, 1);
        if (tid == 0) output[blockIdx.x] = val;
    }
}
```

### CUB Block Reduction (Preferred)

```cuda
#include <cub/cub.cuh>

__global__ void cubReduce(float* input, float* output, int N) {
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (i < N) ? input[i] : 0.0f;
    float blockSum = BlockReduce(temp).Sum(val);

    if (threadIdx.x == 0) atomicAdd(output, blockSum);
}
```

## Tiling & Loop Unrolling

### Manual Loop Unrolling

```cuda
// Compiler hint to unroll
#pragma unroll
for (int k = 0; k < TILE_SIZE; k++) {
    sum += A[row][k] * B[k][col];
}

// Partial unroll with explicit count
#pragma unroll 4
for (int k = 0; k < N; k++) {
    sum += data[k];
}
```

### Thread Coarsening

Each thread processes multiple elements to amortize launch overhead:

```cuda
__global__ void coarsened(float* data, float* out, int N) {
    int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;  // 4 elements per thread

    float a = (idx < N)               ? data[idx] : 0;
    float b = (idx + blockDim.x < N)  ? data[idx + blockDim.x] : 0;
    float c = (idx + 2*blockDim.x < N)? data[idx + 2*blockDim.x] : 0;
    float d = (idx + 3*blockDim.x < N)? data[idx + 3*blockDim.x] : 0;

    if (idx < N)               out[idx] = a * 2;
    if (idx + blockDim.x < N)  out[idx + blockDim.x] = b * 2;
    if (idx + 2*blockDim.x < N)out[idx + 2*blockDim.x] = c * 2;
    if (idx + 3*blockDim.x < N)out[idx + 3*blockDim.x] = d * 2;
}
```

## Instruction Throughput

### Use Intrinsics for Speed

```cuda
// Fast math intrinsics (less precise, much faster)
float r = __fdividef(a, b);     // fast divide
float s = __sinf(x);            // fast sine
float c = __cosf(x);            // fast cosine
float e = __expf(x);            // fast exp
float l = __logf(x);            // fast log
float sq = __fsqrt_rn(x);       // fast sqrt (round-to-nearest)

// Compile with --use_fast_math for automatic replacement
// Equivalent to: --ftz=true --prec-div=false --prec-sqrt=false --fmad=true
```

### Prefer Single Precision

Double precision throughput is 1/2 to 1/32 of single precision (depends on architecture):

```cuda
// BAD: accidental double precision
float x = 1.0;   // 1.0 is double! promotes expression to double
float y = sin(x); // sin() is double, use sinf()

// GOOD: explicitly single precision
float x = 1.0f;
float y = sinf(x);
```

### Fused Multiply-Add (FMA)

```cuda
// FMA: a*b + c in one instruction, one rounding
float result = fmaf(a, b, c);   // single precision
double result = fma(a, b, c);   // double precision
// nvcc enables FMA by default (--fmad=true)
```

### Integer Division and Modulo

Integer division and modulo are expensive (~20 cycles). When the divisor is a power of 2:

```cuda
// BAD: expensive division
int q = idx / 32;
int r = idx % 32;

// GOOD: bit operations (compiler may do this, but be explicit)
int q = idx >> 5;
int r = idx & 31;
```

## Occupancy Tuning

### Register Pressure

Limit registers per thread to increase occupancy:

```cuda
// Compile-time limit
__global__ void __launch_bounds__(256, 2) myKernel(...) {
    // maxThreadsPerBlock=256, minBlocksPerMultiprocessor=2
    // Compiler will try to limit register usage accordingly
}
```

```bash
# Check register usage
nvcc --resource-usage -o kernel kernel.cu
# Output: Used 32 registers, 4096 bytes smem, ...
```

### Shared Memory Configuration

On some architectures, L1 cache and shared memory share the same on-chip memory:

```cuda
// Prefer more shared memory
cudaFuncSetAttribute(myKernel,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    cudaSharedmemCarveoutMaxShared);

// Or prefer more L1 cache
cudaFuncSetAttribute(myKernel,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    cudaSharedmemCarveoutMaxL1);
```

## Grid-Stride Loop Pattern

Single kernel handles arbitrary data sizes, supports persistent threads:

```cuda
__global__ void gridStrideKernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {
        data[i] = processElement(data[i]);
    }
}

// Launch with any grid size -- works for any N
int blockSize = 256;
int gridSize = min((N + blockSize - 1) / blockSize, maxBlocks);
gridStrideKernel<<<gridSize, blockSize>>>(data, N);
```

Benefits: reuses threads (amortizes launch cost), works for N >> grid size, compatible with cooperative groups.

## Scan (Prefix Sum)

### Block-Level Scan with CUB

```cuda
#include <cub/cub.cuh>

__global__ void blockScan(int* data, int* output, int N) {
    typedef cub::BlockScan<int, 256> BlockScan;
    __shared__ typename BlockScan::TempStorage temp;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int val = (i < N) ? data[i] : 0;

    int inclusive;
    BlockScan(temp).InclusiveSum(val, inclusive);

    if (i < N) output[i] = inclusive;
}
```

### Device-Wide Scan with CUB

```cuda
#include <cub/device/device_scan.cuh>

void* d_temp = nullptr;
size_t tempBytes = 0;
cub::DeviceScan::InclusiveSum(d_temp, tempBytes, d_in, d_out, N);
cudaMalloc(&d_temp, tempBytes);
cub::DeviceScan::InclusiveSum(d_temp, tempBytes, d_in, d_out, N);
cudaFree(d_temp);
```

## Kernel Fusion

Combine multiple kernels to avoid intermediate global memory reads/writes:

```cuda
// BAD: three kernels, two intermediate buffers
normalize<<<g,b>>>(input, temp1, N);
scale<<<g,b>>>(temp1, temp2, N, factor);
clamp<<<g,b>>>(temp2, output, N, lo, hi);

// GOOD: fused into one kernel
__global__ void normalizeScaleClamp(float* in, float* out, int N,
                                     float factor, float lo, float hi) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float val = in[i] / norm;  // normalize
        val *= factor;              // scale
        out[i] = fminf(fmaxf(val, lo), hi);  // clamp
    }
}
```
