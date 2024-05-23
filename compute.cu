#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda_runtime.h>

__global__ void compAcc(vector3 *hPos, vector3 *accels, double *mass){
	int i = (blockDim.x * blockIdx.x) + threadIdx.x;
	int j = (blockDim.y * blockIdx.y) + threadIdx.y;

	if(i < NUMENTITIES && j < NUMENTITIES){
		if (i != j){
			vector3 dist;
			for (int k = 0; k < 3; k++) {
				dist[k] = hPos[i][k] - d_hPos[j][k];
			}

			double mag_sq = dist[0] * dist[0] + dist[1] * dist[1] + dist[2] * dist[2];
			double mag = sqrt(mag_sq);
			double force = (-GRAV_CONSTANT * mass[j]) / mag_sq;
			
			for (int k = 0; k < 3; k++) {
				accels[i * NUMENTITIES + j][k] = force * dist[k] / mag;
			}
		}
		else {
			for (int k = 0; k < 3; k++) {
				accels[i * NUMENTITIES + j][k] = 0.0;
			}
		}
	}
} 

__global__ void sum(vector3 *hPos, vector3 *hVel, vector3 *accels, vector3 *accel_sum){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < NUMENTITIES){
		FILL_VECTOR(accel_sum[i], 0, 0, 0);
		for (int j = 0; j < NUMENTITIES; j++){
			for (int k = 0; k < 3; k++){
				accel_sum[i][k] += accels[(i * NUMENTITIES) + j][k];
			}
		}
		for (int k = 0; k < 3; k++){
			hVel[i][k] += accel_sum[i][k] * INTERVAL;
			hPos[i][k] = hVel[i][k] * INTERVAL;
		}

	}
}

void compute(){
	vector3 *dhPos, *dhVel, *daccel, *dsum;
	double *dmass;
	int block = ceilf(NUMENTITIES / 16.0f);
	int thread = ceilf(NUMENTITIES / (float) block);
	dim3 gridDim(block, block, 1);
	dim3 blockDim(thread, thread, 1);
	
	cudaMalloc((void**) &dhPos, sizeof(vector3) * NUMENTITIES);
	cudaMalloc((void**) &dhVel, sizeof(vector3) * NUMENTITIES);
	cudaMalloc((void**) &daccel, sizeof(vector3) * NUMENTITIES);
	cudaMalloc((void**) &dsum, sizeof(vector3) * NUMENTITIES);
	cudaMalloc((void**) &dmass, sizeof(double) * NUMENTITIES);

	cudaMemcpy(dhPos, hPos, sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(dhVel, hVel, sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(dmass, mass, sizeof(double)*NUMENTITIES, cudaMemcpyHostToDevice);
	
	compAcc<<<gridDim, blockDim>>>(dhPos, daccel, dmass);
	cudaDeviceSynchronize();

	sum<<<gridDim.x, blockDim.x>>>(daccel, dsum, dhPos, dhVel);
	cudaDeviceSynchronize();

	cudaMemcpy(hPos, dhPos, sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);
	cudaMemcpy(hVel, dhVel, sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);

	cudaFree(dhPos);
	cudaFree(dhVel);
	cudaFree(dmass);
	cudaFree(daccel);

}
