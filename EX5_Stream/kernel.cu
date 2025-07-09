#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <random>

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
	const int threadsPerBlock = 256;
	const int size = 1024 * 1024 * 256;

	int* a = nullptr;
	int* b = nullptr;
	int* c = nullptr;
	int* cc = nullptr;

	// pinned-memory
	cudaMallocHost(&a, sizeof(int) * size);
	cudaMallocHost(&b, sizeof(int) * size);
	cudaMallocHost(&c, sizeof(int) * size);
	cudaMallocHost(&cc, sizeof(int) * size);

	for (int i = 0; i < size; i++)
	{
		a[i] = std::rand() % 10;
		b[i] = rand() % 10;
		cc[i] = a[i] + b[i];
	}

	// Add vectors using stream.
	{
		cudaStream_t stream;
		cudaStreamCreate(&stream);

		int* dev_a = nullptr;
		int* dev_b = nullptr;
		int* dev_c = nullptr;

		cudaMalloc((void**)&dev_a, size * sizeof(int)); // input a
		cudaMalloc((void**)&dev_b, size * sizeof(int)); // input b
		cudaMalloc((void**)&dev_c, size * sizeof(int)); // output c

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start, 0);

		cudaMemcpyAsync(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice, stream); // 비동기적으로 복사 복사
		cudaMemcpyAsync(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice, stream);

		int numBlocks = int(ceil(float(size) / threadsPerBlock)); // 블럭 여러 개 사용
		addKernel << <numBlocks, threadsPerBlock, 0, stream >> > (dev_a, dev_b, dev_c, size);

		cudaMemcpyAsync(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost, stream);

		cudaEventRecord(stop, 0);  // 끝나는 시간 기록
		cudaDeviceSynchronize();   // kernel이 끝날때까지 대기 (동기화)

		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop); // 걸린 시간 계산
		std::cout << "Time elapsed: " << milliseconds << " ms" << std::endl;

		for (int i = 0; i < size; i++)
		{
			if (cc[i] - c[i] > 1e-4)
			{
				std::cout << "Wrong result" << std::endl;
				goto EXIT;
			}
		}

		std::cout << "Correct" << std::endl;

	EXIT:
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		cudaFree(dev_c);
		cudaFree(dev_a);
		cudaFree(dev_b);

		cudaFreeHost(a);
		cudaFreeHost(b);
		cudaFreeHost(cc);
		cudaFreeHost(c);

		cudaDeviceReset();
	}

	return 0;
}

