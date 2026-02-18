# Modern CUDA Features

## Thrust (High-Level Parallel Algorithms)

Thrust is a C++ parallel algorithms library resembling the STL. Prefer Thrust for standard operations before writing custom kernels.

### Core Algorithms

```cuda
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/copy.h>

// Device vector -- RAII memory management
thrust::device_vector<float> d_vec(N);
thrust::device_vector<float> d_out(N);

// Copy from host
thrust::host_vector<float> h_vec(N);
std::iota(h_vec.begin(), h_vec.end(), 0.0f);
d_vec = h_vec;  // automatic H->D copy

// Sort
thrust::sort(d_vec.begin(), d_vec.end());
thrust::sort(d_vec.begin(), d_vec.end(), thrust::greater<float>());

// Reduce
float sum = thrust::reduce(d_vec.begin(), d_vec.end(), 0.0f, thrust::plus<float>());
float maxVal = thrust::reduce(d_vec.begin(), d_vec.end(), -FLT_MAX, thrust::maximum<float>());

// Transform (element-wise operation)
thrust::transform(d_vec.begin(), d_vec.end(), d_out.begin(),
                  [] __device__ (float x) { return x * x; });

// Transform-reduce (map then reduce -- fused)
float sumSquares = thrust::transform_reduce(d_vec.begin(), d_vec.end(),
    [] __device__ (float x) { return x * x; },
    0.0f, thrust::plus<float>());

// Scan (prefix sum)
thrust::inclusive_scan(d_vec.begin(), d_vec.end(), d_out.begin());
thrust::exclusive_scan(d_vec.begin(), d_vec.end(), d_out.begin(), 0.0f);

// Copy-if (stream compaction)
auto end = thrust::copy_if(d_vec.begin(), d_vec.end(), d_out.begin(),
    [] __device__ (float x) { return x > 0.5f; });
int count = end - d_out.begin();

// Unique
auto newEnd = thrust::unique(d_vec.begin(), d_vec.end());

// Count
int n = thrust::count_if(d_vec.begin(), d_vec.end(),
    [] __device__ (float x) { return x > 0.0f; });
```

### Thrust with Custom Streams

```cuda
cudaStream_t stream;
cudaStreamCreate(&stream);
auto policy = thrust::cuda::par.on(stream);

thrust::sort(policy, d_vec.begin(), d_vec.end());
thrust::reduce(policy, d_vec.begin(), d_vec.end());
```

### Fancy Iterators

```cuda
// Counting iterator (virtual sequence 0, 1, 2, ...)
thrust::counting_iterator<int> first(0);
thrust::counting_iterator<int> last(N);

// Transform iterator (lazy element-wise transform)
auto squares = thrust::make_transform_iterator(first,
    [] __device__ (int x) { return x * x; });
float sumOfSquares = thrust::reduce(squares, squares + N);

// Zip iterator (iterate multiple arrays in lockstep)
auto zipped = thrust::make_zip_iterator(
    thrust::make_tuple(d_x.begin(), d_y.begin()));

// Constant iterator
thrust::constant_iterator<float> ones(1.0f);
float count = thrust::reduce(ones, ones + N);  // = N
```

## CUB (CUDA UnBound)

Lower-level building blocks than Thrust. Use when you need more control or custom kernels:

### Device-Wide Operations

```cuda
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>

// Pattern: query temp storage, allocate, execute
void* d_temp = nullptr;
size_t tempBytes = 0;

// Radix sort
cub::DeviceRadixSort::SortKeys(d_temp, tempBytes, d_keys_in, d_keys_out, N);
cudaMalloc(&d_temp, tempBytes);
cub::DeviceRadixSort::SortKeys(d_temp, tempBytes, d_keys_in, d_keys_out, N);

// Sort key-value pairs
cub::DeviceRadixSort::SortPairs(d_temp, tempBytes,
    d_keys_in, d_keys_out, d_vals_in, d_vals_out, N);

// Reduce
cub::DeviceReduce::Sum(d_temp, tempBytes, d_in, d_out, N);
cub::DeviceReduce::Max(d_temp, tempBytes, d_in, d_out, N);
cub::DeviceReduce::ArgMax(d_temp, tempBytes, d_in, d_out, N);

// Select (compaction)
cub::DeviceSelect::If(d_temp, tempBytes, d_in, d_out, d_numSelected, N, selector);
cub::DeviceSelect::Unique(d_temp, tempBytes, d_in, d_out, d_numSelected, N);
```

### Block-Level Primitives

```cuda
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>

__global__ void kernel(float* input, float* output, int N) {
    // Block-level reduce
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage reduceTemp;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (i < N) ? input[i] : 0;
    float blockSum = BlockReduce(reduceTemp).Sum(val);

    // Block-level scan
    typedef cub::BlockScan<float, 256> BlockScan;
    __shared__ typename BlockScan::TempStorage scanTemp;
    float inclusive;
    BlockScan(scanTemp).InclusiveSum(val, inclusive);

    if (i < N) output[i] = inclusive;
}
```

### Warp-Level Primitives

```cuda
#include <cub/warp/warp_reduce.cuh>
#include <cub/warp/warp_scan.cuh>

__global__ void kernel(float* data, float* warpSums, int N) {
    typedef cub::WarpReduce<float> WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp[8];  // 8 warps per block

    int warpId = threadIdx.x / 32;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (i < N) ? data[i] : 0;
    float warpSum = WarpReduce(temp[warpId]).Sum(val);
}
```

## CUDA C++ Standard Library (libcu++)

Modern C++ in device code (CUDA 12+):

```cuda
#include <cuda/std/atomic>
#include <cuda/std/barrier>
#include <cuda/std/semaphore>
#include <cuda/std/latch>

// Scoped atomics
__device__ cuda::atomic<int, cuda::thread_scope_block> blockCounter{0};
__device__ cuda::atomic<int, cuda::thread_scope_device> deviceCounter{0};
__device__ cuda::atomic<int, cuda::thread_scope_system> systemCounter{0};

// Barrier
__global__ void kernel() {
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;
    if (threadIdx.x == 0) init(&bar, blockDim.x);
    __syncthreads();

    // Phase 1: produce data
    produce(threadIdx.x);
    bar.arrive_and_wait();

    // Phase 2: consume data
    consume(threadIdx.x);
}
```

## Tensor Cores (via WMMA or cuBLAS)

### WMMA API (Warp-Level Matrix Multiply)

```cuda
#include <mma.h>
using namespace nvcuda;

__global__ void wmmaKernel(half* A, half* B, float* C, int M, int N, int K) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // Load matrix tiles
    wmma::load_matrix_sync(a_frag, A + warpRow * K, K);
    wmma::load_matrix_sync(b_frag, B + warpCol, N);

    // Tensor Core matrix multiply-accumulate
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store result
    wmma::store_matrix_sync(C + warpRow * N + warpCol, c_frag, N, wmma::mem_row_major);
}
```

**Prefer cuBLAS** for production GEMM -- it uses Tensor Cores automatically:

```cuda
#include <cublas_v2.h>

cublasHandle_t handle;
cublasCreate(&handle);
cublasSetStream(handle, stream);

float alpha = 1.0f, beta = 0.0f;
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);

cublasDestroy(handle);
```

## cuBLAS, cuFFT, cuSPARSE, cuRAND

Always prefer these optimized libraries over custom kernels when possible:

```cuda
// cuFFT -- Fast Fourier Transform
#include <cufft.h>
cufftHandle plan;
cufftPlan1d(&plan, N, CUFFT_C2C, 1);
cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
cufftDestroy(plan);

// cuRAND -- Random Number Generation
#include <curand.h>
curandGenerator_t gen;
curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
curandGenerateUniform(gen, d_data, N);
curandDestroyGenerator(gen);
```

## Architecture-Specific Features

### Hopper (SM 90) -- H100

- **Thread Block Clusters:** groups of thread blocks that can cooperate
- **Asynchronous TMA (Tensor Memory Accelerator):** hardware-accelerated bulk data movement
- **Distributed shared memory:** shared memory accessible across blocks in a cluster

```cuda
// Cluster launch (CUDA 12+)
cudaLaunchConfig_t config = {};
config.gridDim = grid;
config.blockDim = block;

cudaLaunchAttribute attrs[1];
attrs[0].id = cudaLaunchAttributeClusterDimension;
attrs[0].val.clusterDim = {2, 1, 1};  // 2 blocks per cluster
config.attrs = attrs;
config.numAttrs = 1;

cudaLaunchKernelEx(&config, myKernel, args...);
```

### Blackwell (SM 100) -- B200

- **Fifth-gen Tensor Cores:** FP4 and FP6 support
- **Extended L2 cache:** up to 96MB
- **Enhanced TMA:** improved async copy capabilities
- Follow NVIDIA Blackwell Tuning Guide for architecture-specific optimizations

### Best Practice: Architecture Portability

```cuda
// Query architecture at runtime
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

if (prop.major >= 9) {
    // Hopper+ features (clusters, TMA)
} else if (prop.major >= 8) {
    // Ampere features (async copy, reduced precision TF32)
} else {
    // Fallback path
}
```

## CUDA Python Interop

### CuPy (NumPy on GPU)

```python
import cupy as cp

# Array operations on GPU
a = cp.array([1, 2, 3], dtype=cp.float32)
b = cp.sum(a)

# Raw kernel
kernel = cp.RawKernel(r'''
extern "C" __global__
void myKernel(float* x, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) x[i] *= 2;
}
''', 'myKernel')
kernel((grid,), (block,), (d_x, n))
```

### PyTorch Custom CUDA Extensions

```python
# setup.py
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    ext_modules=[CUDAExtension('my_cuda', ['my_cuda.cu'])],
    cmdclass={'build_ext': BuildExtension}
)
```
