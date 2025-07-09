#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <vector>
#include <random>

// Error check macro.
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d in %s: %s\n", \
                    __FILE__, __LINE__, __func__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

__global__ void addKernel(const int* a, const int* b, int* c, int size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < size)
	{
		c[i] = a[i] + b[i];
	}
}

int main()
{
	const int threadsPerBlock = 1024;
	const int size = 1024 * 1024 * 512;
	const int numSplits = 8;
	const int splitSize = size / numSplits;

	int* a = nullptr;
	int* b = nullptr;
	int* c = nullptr;

	cudaMallocHost(&a, sizeof(int) * size); // pinned-memory
	cudaMallocHost(&b, sizeof(int) * size);
	cudaMallocHost(&c, sizeof(int) * size);

	for (int i = 0; i < size; i++)
	{
		a[i] = rand() % 10;
		b[i] = rand() % 10;
	}

	// Add large vector.
	{
		cudaStream_t stream;
		cudaStreamCreate(&stream);

		int* dev_a = nullptr;
		int* dev_b = nullptr;
		int* dev_c = nullptr;

		CUDA_CHECK(cudaMalloc((void**)&dev_a, splitSize * sizeof(int)));
		CUDA_CHECK(cudaMalloc((void**)&dev_b, splitSize * sizeof(int)));
		CUDA_CHECK(cudaMalloc((void**)&dev_c, splitSize * sizeof(int)));

		cudaEvent_t start, stop;
		CUDA_CHECK(cudaEventCreate(&start));
		CUDA_CHECK(cudaEventCreate(&stop));

		CUDA_CHECK(cudaEventRecord(start, 0));

		for (int s = 0; s < numSplits; s++)
		{
			CUDA_CHECK(cudaMemcpyAsync(dev_a, &a[s * splitSize], splitSize * sizeof(int), cudaMemcpyHostToDevice, stream)); // size -> split_size
			CUDA_CHECK(cudaMemcpyAsync(dev_b, &b[s * splitSize], splitSize * sizeof(int), cudaMemcpyHostToDevice, stream)); // size -> split_size

			int numBlocks = int(ceil(float(splitSize) / threadsPerBlock));
			addKernel << <numBlocks, threadsPerBlock, 0, stream >> > (dev_a, dev_b, dev_c, splitSize);

			CUDA_CHECK(cudaMemcpyAsync(&c[s * splitSize], dev_c, splitSize * sizeof(int), cudaMemcpyDeviceToHost, stream));
		}

		CUDA_CHECK(cudaEventRecord(stop, 0));
		CUDA_CHECK(cudaDeviceSynchronize());

		float milliseconds = 0;
		CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
		std::cout << "Time elapsed: " << milliseconds << " ms" << std::endl;

		for (int i = 0; i < size; i++)
		{
			if (c[i] != a[i] + b[i])
			{
				std::cout << "Wrong result" << std::endl;
				goto EXIT;
			}
		}

		std::cout << "Correct" << std::endl;

	EXIT:
		CUDA_CHECK(cudaEventDestroy(start));
		CUDA_CHECK(cudaEventDestroy(stop));

		CUDA_CHECK(cudaFree(dev_c));
		CUDA_CHECK(cudaFree(dev_a));
		CUDA_CHECK(cudaFree(dev_b));

		CUDA_CHECK(cudaFreeHost(a));
		CUDA_CHECK(cudaFreeHost(b));
		CUDA_CHECK(cudaFreeHost(c));

		CUDA_CHECK(cudaDeviceReset());
	}


	return 0;
}