#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <iomanip>
#include <vector>

__global__ void addKernel(int* c, const int* a, const int* b, int size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size)
	{
		c[i] = a[i] + b[i];
	}
}

int main()
{
	constexpr int size = 1024 * 1024;
	std::vector<int> a(size);
	std::vector<int> b(size);
	std::vector<int> c(size);
	std::vector<int> cc(size); // A vector for verifying correctness of result.

	for (int i = 0; i < size; ++i)
	{
		a[i] = std::rand() % 100;
		b[i] = std::rand() % 100;
		cc[i] = a[i] + b[i];
	}

	// Add vectors in parallel.
	{
		// The code that checks cudaStatus is omitted for brevity.
		int* dev_a = nullptr;
		int* dev_b = nullptr;
		int* dev_c = nullptr;
		cudaError_t cudaStatus;

		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaStatus = cudaSetDevice(0);

		// Allocate GPU buffers for three vectors (two input, one output).
		cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
		cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
		cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));

		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(dev_a, a.data(), size * sizeof(int), cudaMemcpyHostToDevice);
		cudaStatus = cudaMemcpy(dev_b, b.data(), size * sizeof(int), cudaMemcpyHostToDevice);

		// Launch a kernel on the GPU with one thread for each element.
		int numThreadsPerBlock = 256;
		int numBlocks = int(std::ceil(float(size) / numThreadsPerBlock));
		addKernel << <numBlocks, numThreadsPerBlock >> > (dev_c, dev_a, dev_b, size);

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();

		// Copy output vector from GPU buffer to host memory.
		cudaStatus = cudaMemcpy(c.data(), dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

		cudaFree(dev_c);
		cudaFree(dev_a);
		cudaFree(dev_b);

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		cudaStatus = cudaDeviceReset();
	}

	bool bCorrect = true;
	for (int i = 0; i < size; ++i)
	{
		bCorrect &= c[i] == cc[i];
	}

	std::string resultPhrase = bCorrect ? "Correct!" : "Wrong!";
	std::cout << resultPhrase << std::endl;

	return 0;
}