#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <iomanip>
#include <vector>

struct Matrix
{
	int Height = 0;
	int Width = 0;
	float* Data = nullptr;
};

void matMulCPU(const Matrix& a, const Matrix& b, Matrix* out)
{
	int M = a.Height;
	int K = a.Width;
	int N = b.Width;
	for (int row = 0; row < M; ++row)
	{
		for (int col = 0; col < N; ++col) {
			float v = 0.0f;
			for (int e = 0; e < K; e++)
			{
				v += a.Data[row * a.Width + e] * b.Data[e * b.Width + col];
			}
			out->Data[row * out->Width + col] = v;
		}
	}
}

__global__ void matMulKernel(Matrix a, Matrix b, Matrix out)
{
	float v = 0.0f;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	for (int e = 0; e < a.Width; ++e)
	{
		v += a.Data[row * a.Width + e] * b.Data[e * b.Width + col];
	}
	out.Data[row * out.Width + col] = v;
}

int main()
{
	const int M = 1024 * 2;
	const int N = 1024 * 1;
	const int K = 256;

	Matrix matA{ M, K };
	Matrix matB{ K, N };
	Matrix matC{ M, N };
	Matrix matCC{ M, N };

	matA.Data = new float[matA.Width * matA.Height];
	for (int i = 0; i < matA.Width * matA.Height; i++) matA.Data[i] = 0.1f * (float)(std::rand() % 10);
	matB.Data = new float[matB.Width * matB.Height];
	for (int i = 0; i < matB.Width * matB.Height; i++) matB.Data[i] = 0.1f * (float)(std::rand() % 10);
	matC.Data = new float[matC.Width * matC.Height];
	for (int i = 0; i < matC.Width * matC.Height; i++) matC.Data[i] = 0.0f;

	matCC.Data = new float[matCC.Width * matCC.Height];
	matMulCPU(matA, matB, &matCC);

	// Sum vectors in parallel.
	{
		Matrix devMatA{ M,K };
		Matrix devMatB{ K,N };
		Matrix devMatC{ M,N };
		cudaError_t cudaStatus;

		cudaStatus = cudaSetDevice(0);

		cudaStatus = cudaMalloc((void**)&devMatA.Data, devMatA.Width * devMatA.Height * sizeof(float));
		cudaStatus = cudaMalloc((void**)&devMatB.Data, devMatB.Width * devMatB.Height * sizeof(float));
		cudaStatus = cudaMalloc((void**)&devMatC.Data, devMatC.Width * devMatC.Height * sizeof(float));

		cudaStatus = cudaMemcpy(devMatA.Data, matA.Data, devMatA.Width * devMatA.Height * sizeof(float), cudaMemcpyHostToDevice);
		cudaStatus = cudaMemcpy(devMatB.Data, matB.Data, devMatB.Width * devMatB.Height * sizeof(float), cudaMemcpyHostToDevice);
		cudaStatus = cudaMemcpy(devMatC.Data, matC.Data, devMatC.Width * devMatC.Height * sizeof(float), cudaMemcpyHostToDevice);

		dim3 dimBlock(32, 32, 1);
		dim3 dimGrid(devMatC.Width / dimBlock.x, devMatC.Height / dimBlock.y); // Assert no remainder

		matMulKernel << <dimGrid, dimBlock >> > (devMatA, devMatB, devMatC);

		cudaStatus = cudaDeviceSynchronize();

		cudaStatus = cudaMemcpy(matC.Data, devMatC.Data, devMatC.Width * devMatC.Height * sizeof(float), cudaMemcpyDeviceToHost);

		cudaFree(devMatA.Data);
		cudaFree(devMatB.Data);
		cudaFree(devMatC.Data);

		cudaStatus = cudaDeviceReset();
	}

	// Check result.
	bool bCorrect = true;
	for (int i = 0; i < matC.Width * matC.Height; i++)
	{
		bCorrect &= (1e-4 > std::fabsf(matC.Data[i] - matCC.Data[i]));
	}

	std::string resultPhrase = bCorrect ? "Correct!" : "Wrong!";
	std::cout << resultPhrase << std::endl;

	// Free memory.
	delete[] matA.Data;
	delete[] matB.Data;
	delete[] matC.Data;
	delete[] matCC.Data;

	return 0;
}