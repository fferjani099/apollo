#ifndef GPU_INTERCEPT_H
#define GPU_INTERCEPT_H

#include <cuda_runtime.h>
#include <dlfcn.h>
#include <pthread.h>
#include <queue>
#include <mutex>
#include <cassert>
#include <iostream>

// Data structures for different CUDA calls
struct KernelRecord {
    const void* func;
    dim3 gridDim;
    dim3 blockDim;
    void** args;
    size_t sharedMem;
    cudaStream_t stream;
};

struct MemcpyRecord {
	void* dst;
	const void* src;
	size_t count;
	enum cudaMemcpyKind kind;
	cudaStream_t stream;
	bool async;
};

struct MemsetRecord {

	void* devPtr;
	int value;
	size_t count;
	cudaStream_t stream;
	bool async;

};

struct MallocRecord {
    void** devPtr;
    size_t size;
};

struct FreeRecord {
    void* devPtr;
};

// Function pointers to original CUDA functions
static cudaError_t (*original_cudaMalloc)(void**, size_t) = nullptr;
static cudaError_t (*original_cudaFree)(void*) = nullptr;
static cudaError_t (*original_cudaLaunchKernel)(const void*, dim3, dim3, void**, size_t, cudaStream_t) = nullptr;

// Queues for intercepted CUDA calls
std::queue<KernelRecord> kernelQueue;
std::queue<MallocRecord> mallocQueue;
std::queue<FreeRecord> freeQueue;
std::mutex queueMutex;

// Scheduler functions
void scheduleCudaMalloc(MallocRecord &record);
void scheduleCudaFree(FreeRecord &record);
void scheduleCudaLaunchKernel(KernelRecord &record);

// Helper functions
void initOriginalFunctions();

#endif // GPU_INTERCEPT_H
