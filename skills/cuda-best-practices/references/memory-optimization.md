# CUDA Memory Optimization

## Memory Hierarchy Overview

| Memory        | Scope        | Speed     | Size          | Cached | Notes                         |
|---------------|-------------|-----------|---------------|--------|-------------------------------|
| Registers     | Per-thread   | Fastest   | ~255 per thread | N/A  | Compiler-managed              |
| Shared        | Per-block    | ~100x global | Up to 228KB/SM | N/A | Programmer-managed            |
| L1 Cache      | Per-SM       | Fast      | Up to 228KB/SM | Yes  | Combined with shared memory   |
| L2 Cache      | Device-wide  | Medium    | Up to 96MB    | Yes   | All global/local access       |
| Constant      | Device-wide  | Fast (broadcast) | 64KB  | Yes   | Read-only, set from host      |
| Texture       | Device-wide  | Fast      | Limited       | Yes   | Spatial locality optimized    |
| Global (VRAM) | Device-wide  | Slowest   | Up to 80GB+   | L1/L2 | Main device memory            |
| Local         | Per-thread   | Same as global | Spill space | L1/L2 | Register spill + large arrays |

## Global Memory Coalescing

### Rules for Coalesced Access

Optimal coalescing: all threads in a warp access addresses within the same 128-byte cache line (or two 32-byte sectors for L2).

```cuda
// PERFECT coalescing: thread i reads element i
__global__ void readCoalesced(float* data, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = data[i] * 2.0f;
}

// OFFSET access: slight penalty, still reasonable
__global__ void readOffset(float* data, float* out, int N, int offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (i < N) out[i - offset] = data[i];
}

// STRIDED access: TERRIBLE performance
// stride of 2 = 50% bandwidth waste; stride of 32 = 3% efficiency
__global__ void readStrided(float* data, float* out, int N, int stride) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    if (i < N) out[i / stride] = data[i];
}
```

### SoA vs AoS Transformation

```cuda
// AoS: Array of Structures -- BAD for GPU coalescing
struct ParticleAoS {
    float3 position;
    float3 velocity;
    float mass;
};
// Thread 0 reads particle[0].position.x at offset 0
// Thread 1 reads particle[1].position.x at offset 28 bytes -- strided!

// SoA: Structure of Arrays -- GOOD for GPU coalescing
struct ParticlesSoA {
    float* pos_x;   // pos_x[0], pos_x[1], pos_x[2]... contiguous
    float* pos_y;
    float* pos_z;
    float* vel_x;
    float* vel_y;
    float* vel_z;
    float* mass;
};
// Thread 0 reads pos_x[0], Thread 1 reads pos_x[1] -- coalesced!
```

### Vectorized Loads

Use `float2`, `float4`, `int4` for wider loads (single instruction loads 8/16 bytes):

```cuda
__global__ void vectorizedCopy(float4* dst, const float4* src, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = src[i];  // 16-byte load in one instruction
}

// Reinterpret cast from float* to float4*
// Requires 16-byte alignment and N divisible by 4
float4* d_data4 = reinterpret_cast<float4*>(d_data);
vectorizedCopy<<<(N/4+255)/256, 256>>>(d_dst4, d_data4, N/4);
```

## Shared Memory

### Tiled Matrix Multiplication

Classic example of using shared memory to reduce global memory traffic:

```cuda
#define TILE 32

__global__ void matMul(float* C, const float* A, const float* B, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < N / TILE; t++) {
        // Coalesced load into shared memory
        As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        __syncthreads();

        // Compute partial dot product from shared memory
        for (int k = 0; k < TILE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}
```

Each element of A and B is loaded from global memory once per tile instead of N times.

### Bank Conflict Avoidance

Shared memory has 32 banks. Bank index = `(address / 4) % 32` for 4-byte words.

```cuda
// CONFLICT: all threads access same bank (stride 32)
__shared__ float data[1024];
float val = data[threadIdx.x * 32];  // bank 0 for all threads!

// NO CONFLICT: consecutive access (stride 1)
float val = data[threadIdx.x];  // thread i -> bank i

// PADDING TRICK for transpose (eliminates conflicts):
__shared__ float tile[32][32];     // column access: bank conflict!
__shared__ float tile[32][32 + 1]; // +1 padding: no conflict

// Broadcast: all threads reading the SAME address is conflict-free
float val = data[0];  // broadcast to entire warp -- no conflict
```

### Dynamic Shared Memory

```cuda
extern __shared__ float sharedMem[];

__global__ void kernel(float* data, int N) {
    // Use offsets to partition dynamic shared memory
    float* arrA = sharedMem;               // first N floats
    float* arrB = sharedMem + N;           // next N floats
    int*   arrC = (int*)(sharedMem + 2*N); // next N ints
    // ...
}

// Launch with shared memory size specified
kernel<<<grid, block, sharedMemBytes>>>(data, N);
```

## Unified Memory

Unified memory (`cudaMallocManaged`) provides a single pointer accessible from both CPU and GPU. The driver handles page migration automatically.

```cuda
float* data;
cudaMallocManaged(&data, N * sizeof(float));

// Initialize on host
for (int i = 0; i < N; i++) data[i] = i;

// Use on device -- pages migrate on demand
kernel<<<grid, block>>>(data, N);
cudaDeviceSynchronize();

// Read back on host -- pages migrate back
printf("%f\n", data[0]);

cudaFree(data);
```

### Prefetching for Performance

Without hints, unified memory relies on page faults (expensive). Prefetch explicitly:

```cuda
cudaMallocManaged(&data, size);
initOnHost(data, N);

// Prefetch to GPU before kernel launch
cudaMemPrefetchAsync(data, size, deviceId, stream);
kernel<<<grid, block, 0, stream>>>(data, N);

// Prefetch back to CPU before host access
cudaMemPrefetchAsync(data, size, cudaCpuDeviceId, stream);
cudaStreamSynchronize(stream);
useOnHost(data, N);
```

### Advice Hints

```cuda
// Tell the runtime the data is mostly read
cudaMemAdvise(data, size, cudaMemAdviseSetReadMostly, deviceId);

// Set preferred location
cudaMemAdvise(data, size, cudaMemAdviseSetPreferredLocation, deviceId);
cudaMemAdvise(data, size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);

// Set accessor device (avoid page faults, use direct mapping)
cudaMemAdvise(data, size, cudaMemAdviseSetAccessedBy, deviceId);
```

## Pinned (Page-Locked) Memory

Pinned memory enables DMA transfers (faster) and is required for `cudaMemcpyAsync`:

```cuda
float* h_data;
cudaMallocHost(&h_data, size);  // pinned

// Alternative: register existing allocation
float* h_existing = (float*)malloc(size);
cudaHostRegister(h_existing, size, cudaHostRegisterDefault);

// Mapped pinned memory (zero-copy: GPU accesses host memory directly)
float* h_mapped;
cudaHostAlloc(&h_mapped, size, cudaHostAllocMapped);
float* d_mapped;
cudaHostGetDevicePointer(&d_mapped, h_mapped, 0);
kernel<<<grid, block>>>(d_mapped, N);  // GPU reads from host memory
```

**Warning:** Don't over-allocate pinned memory -- it reduces available system memory and can degrade OS performance. Use it only for transfer buffers.

## Memory Pool Allocation

CUDA 11.2+ stream-ordered memory allocation avoids `cudaMalloc`/`cudaFree` overhead:

```cuda
float* d_temp;
cudaMallocAsync(&d_temp, size, stream);
kernel<<<grid, block, 0, stream>>>(d_temp, N);
cudaFreeAsync(d_temp, stream);
// Memory returned to pool immediately, no device sync needed
```

## Texture Memory

Optimized for 2D spatial locality. Use via texture objects:

```cuda
cudaTextureObject_t texObj;
cudaResourceDesc resDesc = {};
resDesc.resType = cudaResourceTypeLinear;
resDesc.res.linear.devPtr = d_data;
resDesc.res.linear.desc = cudaCreateChannelDesc<float>();
resDesc.res.linear.sizeInBytes = size;

cudaTextureDesc texDesc = {};
texDesc.readMode = cudaReadModeElementType;

cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);

__global__ void kernel(cudaTextureObject_t tex, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = tex1Dfetch<float>(tex, i);
}
```

Best for: read-only data with spatial locality (images, lookup tables, interpolated access).

## Atomic Operations

Use when multiple threads must update the same location. Atomics serialize at the address level.

```cuda
// Global memory atomics
atomicAdd(&globalCounter, 1);
atomicMax(&globalMax, localVal);
atomicCAS(&lock, 0, 1);  // compare-and-swap

// Shared memory atomics (faster than global)
__shared__ int blockCount;
if (threadIdx.x == 0) blockCount = 0;
__syncthreads();
atomicAdd(&blockCount, 1);

// Reduce atomic contention: per-warp or per-block reduction first
float warpSum = warpReduceSum(myVal);
if (laneId == 0) atomicAdd(&result, warpSum);  // 32x fewer atomics
```

## Memory Fence Functions

For ensuring visibility across threads (weakly-ordered memory model):

```cuda
__threadfence_block();  // visible to block
__threadfence();        // visible to device
__threadfence_system(); // visible to device + host + peers
```
