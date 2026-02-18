# CUDA Async Concurrency

## Streams

A stream is a sequence of operations (kernel launches, memory copies) that execute in order. Operations in different streams can execute concurrently.

### Stream Basics

```cuda
cudaStream_t stream;
cudaStreamCreate(&stream);

// All three operations execute sequentially within this stream
cudaMemcpyAsync(d_in, h_in, size, cudaMemcpyHostToDevice, stream);
kernel<<<grid, block, 0, stream>>>(d_in, d_out, N);
cudaMemcpyAsync(h_out, d_out, size, cudaMemcpyDeviceToHost, stream);

// Synchronize
cudaStreamSynchronize(stream);  // block host until stream completes
cudaStreamDestroy(stream);
```

### Multi-Stream Pipeline

Overlap copy and compute by splitting work across streams:

```cuda
const int nStreams = 4;
const int chunkSize = N / nStreams;
cudaStream_t streams[nStreams];

for (int i = 0; i < nStreams; i++)
    cudaStreamCreate(&streams[i]);

for (int i = 0; i < nStreams; i++) {
    int offset = i * chunkSize;
    cudaMemcpyAsync(&d_in[offset], &h_in[offset],
                     chunkSize * sizeof(float),
                     cudaMemcpyHostToDevice, streams[i]);
    kernel<<<chunkSize/256, 256, 0, streams[i]>>>(&d_in[offset], &d_out[offset]);
    cudaMemcpyAsync(&h_out[offset], &d_out[offset],
                     chunkSize * sizeof(float),
                     cudaMemcpyDeviceToHost, streams[i]);
}

for (int i = 0; i < nStreams; i++) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
}
```

### Stream Priorities

```cuda
int leastPriority, greatestPriority;
cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);

cudaStream_t highPriority, lowPriority;
cudaStreamCreateWithPriority(&highPriority, cudaStreamNonBlocking, greatestPriority);
cudaStreamCreateWithPriority(&lowPriority, cudaStreamNonBlocking, leastPriority);
```

### Non-Blocking Streams

```cuda
// Default stream (stream 0) implicitly synchronizes with all other streams
// Use cudaStreamNonBlocking to avoid this
cudaStream_t stream;
cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
```

## Events

Events provide timing and inter-stream synchronization:

### Timing with Events

```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);
kernel<<<grid, block, 0, stream>>>(args);
cudaEventRecord(stop, stream);

cudaEventSynchronize(stop);

float ms = 0;
cudaEventElapsedTime(&ms, start, stop);
printf("Kernel took %.3f ms\n", ms);

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

### Inter-Stream Synchronization

```cuda
cudaStream_t producer, consumer;
cudaEvent_t dataReady;
cudaStreamCreate(&producer);
cudaStreamCreate(&consumer);
cudaEventCreate(&dataReady);

// Producer stream generates data
generateKernel<<<g, b, 0, producer>>>(d_data, N);
cudaEventRecord(dataReady, producer);

// Consumer stream waits for data, then processes it
cudaStreamWaitEvent(consumer, dataReady);
processKernel<<<g, b, 0, consumer>>>(d_data, d_output, N);

cudaEventDestroy(dataReady);
cudaStreamDestroy(producer);
cudaStreamDestroy(consumer);
```

## CUDA Graphs

CUDA graphs capture a sequence of operations and replay them with minimal launch overhead. Ideal for repeated workloads.

### Stream Capture

```cuda
cudaGraph_t graph;
cudaGraphExec_t graphExec;
cudaStream_t stream;
cudaStreamCreate(&stream);

// Capture operations into a graph
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

cudaMemcpyAsync(d_in, h_in, size, cudaMemcpyHostToDevice, stream);
kernelA<<<g, b, 0, stream>>>(d_in, d_tmp, N);
kernelB<<<g, b, 0, stream>>>(d_tmp, d_out, N);
cudaMemcpyAsync(h_out, d_out, size, cudaMemcpyDeviceToHost, stream);

cudaStreamEndCapture(stream, &graph);

// Instantiate -- validates and optimizes the graph
cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

// Execute repeatedly with minimal overhead
for (int iter = 0; iter < 1000; iter++) {
    cudaGraphLaunch(graphExec, stream);
}
cudaStreamSynchronize(stream);

// Cleanup
cudaGraphExecDestroy(graphExec);
cudaGraphDestroy(graph);
```

### Explicit Graph Construction

```cuda
cudaGraph_t graph;
cudaGraphCreate(&graph, 0);

// Add nodes manually
cudaGraphNode_t kernelNode;
cudaKernelNodeParams kernelParams = {};
kernelParams.func = (void*)myKernel;
kernelParams.gridDim = grid;
kernelParams.blockDim = block;
kernelParams.kernelParams = kernelArgs;
cudaGraphAddKernelNode(&kernelNode, graph, nullptr, 0, &kernelParams);

// Add dependencies between nodes
cudaGraphAddDependencies(graph, &nodeA, &nodeB, 1);
```

### Graph Update (CUDA 12+)

Modify kernel parameters without recreating the graph:

```cuda
cudaKernelNodeParams newParams = kernelParams;
newParams.kernelParams = newArgs;  // new data pointer, etc.
cudaGraphExecKernelNodeSetParams(graphExec, kernelNode, &newParams);
// Re-launch with updated parameters -- no re-instantiation
cudaGraphLaunch(graphExec, stream);
```

## Cooperative Groups

Flexible thread grouping beyond blocks (CUDA 9+):

```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void kernel(float* data, int N) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // Warp-level operations
    float val = data[blockIdx.x * blockDim.x + threadIdx.x];
    float warpSum = cg::reduce(warp, val, cg::plus<float>());

    // Block-level sync (same as __syncthreads())
    block.sync();

    // Grid-level sync (requires cooperative launch)
    cg::grid_group grid = cg::this_grid();
    grid.sync();  // all blocks synchronize
}

// Cooperative launch required for grid.sync()
void* args[] = { &d_data, &N };
cudaLaunchCooperativeKernel((void*)kernel, grid, block, args);
```

### Multi-Grid Cooperative Launch (Multi-GPU)

```cuda
cg::multi_grid_group multiGrid = cg::this_multi_grid();
multiGrid.sync();  // synchronize across multiple GPUs
```

## Dynamic Parallelism

Kernels launching kernels (compute capability 3.5+):

```cuda
__global__ void parentKernel(float* data, int N) {
    // Launch child kernel from device code
    if (threadIdx.x == 0) {
        childKernel<<<childGrid, childBlock>>>(data, N);
        cudaDeviceSynchronize();  // wait for child (device-side sync)
    }
}
```

Use for: adaptive algorithms (AMR), recursive subdivisions (quicksort), workloads with irregular parallelism.

**Caveats:** higher launch overhead than host-side launch, limited nesting depth, and memory for pending launches is finite.

## Callback and Host Functions in Streams

```cuda
void CUDART_CB myCallback(void* userData) {
    printf("Stream work complete, data=%p\n", userData);
}

// Insert host function into stream -- executes when preceding work completes
cudaLaunchHostFunc(stream, myCallback, userData);
```

## Multi-GPU Programming

### Peer-to-Peer Access

```cuda
int canAccess;
cudaDeviceCanAccessPeer(&canAccess, gpu0, gpu1);
if (canAccess) {
    cudaSetDevice(gpu0);
    cudaDeviceEnablePeerAccess(gpu1, 0);
    // gpu0 kernels can now directly access gpu1 memory
}

// Direct peer copy
cudaMemcpyPeer(d_dst, gpu0, d_src, gpu1, size);
cudaMemcpyPeerAsync(d_dst, gpu0, d_src, gpu1, size, stream);
```
