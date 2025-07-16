#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <iomanip>
#include <vector>

__global__ void atomicSumRedutionKernel(float* inVec, float* outVal)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	atomicAdd(outVal, inVec[i]);
}

__global__ void convergentSumReductionKernel(float* inVec, float* outVal)
{
	for (int stride = blockDim.x / 2; stride >= 1; stride /= 2)
	{
		if (threadIdx.x < stride)
		{
			inVec[threadIdx.x] += inVec[threadIdx.x + stride];
		}
		__syncthreads();
	}

	if (threadIdx.x == 0)
	{
		*outVal = inVec[0];
	}
}

__global__ void sharedMemorySumReductionKernel(float* inVec, float* outVal)
{
	extern __shared__ float inVecShared[];

	int stride = blockDim.x / 2;
	if (threadIdx.x < stride)
	{
		inVecShared[threadIdx.x] = inVec[threadIdx.x] + inVec[threadIdx.x + stride];
	}

	for (stride /= 2; stride >= 1; stride /= 2)
	{
		__syncthreads();
		if (threadIdx.x < stride)
		{
			inVecShared[threadIdx.x] += inVecShared[threadIdx.x + stride];
		}
	}

	if (threadIdx.x == 0)
	{
		*outVal = inVecShared[0];
	}
}

__global__ void segmentedSumReductionKernel(float* inVec, float* outVal)
{
	extern __shared__ float inVecShared[];

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int t = threadIdx.x;

	int stride = blockDim.x / 2;
	if (t < stride)
	{
		inVecShared[t] = inVec[i] + inVec[i + stride];
	}

	for (stride /= 2; stride >= 1; stride /= 2)
	{
		__syncthreads();
		if (t < stride)
		{
			inVecShared[t] += inVecShared[t + stride];
		}
	}

	if (t == 0)
	{
		atomicAdd(outVal, inVecShared[0]);
	}
}

int main()
{
	constexpr int size = 1024 * 1024;
	std::vector<float> input(size);
	float output = 0.0f;
	float outputCorrect = 0.0f; // A value for verifying correctness of result.

	for (int i = 0; i < size; ++i)
	{
		input[i] = 0.01f * (float)(std::rand() % 10);
		outputCorrect += input[i];
	}

	// Sum vectors in parallel.
	{
		float* devInput = nullptr;
		float* devOutput = nullptr;
		cudaError_t cudaStatus;

		cudaStatus = cudaSetDevice(0);

		cudaStatus = cudaMalloc((void**)&devInput, size * sizeof(float));
		cudaStatus = cudaMalloc((void**)&devOutput, sizeof(float));

		cudaStatus = cudaMemcpy(devInput, input.data(), size * sizeof(float), cudaMemcpyHostToDevice);

		int numThreadsPerBlock = 1024;
		int numBlocks = int(std::ceil(float(size) / numThreadsPerBlock));

		// atomicSumRedutionKernel << <numBlocks, numThreadsPerBlock >> > (devInput, devOutput);

		// convergentSumReductionKernel << <1, numThreadsPerBlock >> > (devInput, devOutput);

		// sharedMemorySumReductionKernel << < numBlocks, numThreadsPerBlock, numThreadsPerBlock * sizeof(float) >> > (devInput, devOutput);

		segmentedSumReductionKernel << < numBlocks, numThreadsPerBlock, numThreadsPerBlock * sizeof(float) >> > (devInput, devOutput);

		cudaStatus = cudaDeviceSynchronize();

		cudaStatus = cudaMemcpy(&output, devOutput, sizeof(float), cudaMemcpyDeviceToHost);

		cudaFree(devInput);
		cudaFree(devOutput);

		cudaStatus = cudaDeviceReset();
	}

	std::cout << "CPU Result : " << outputCorrect << std::endl;
	std::cout << "GPU Result : " << output << std::endl;

	return 0;
}