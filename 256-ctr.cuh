// System includes
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <ctime>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
//#include <helper_functions.h>
//#include <helper_cuda.h>

#include <device_launch_parameters.h>
//#include <device_functions.h>

// Custom header 
//#include "kernel.h"


// Key expansion from given key set, populate rk[52]
__host__ void keyExpansion256(u32* key, u32* rk) {

	u32 rk0, rk1, rk2, rk3, rk4, rk5, rk6, rk7;
	rk0 = key[0];
	rk1 = key[1];
	rk2 = key[2];
	rk3 = key[3];
	rk4 = key[4];
	rk5 = key[5];
	rk6 = key[6];
	rk7 = key[7];

	rk[0] = rk0;
	rk[1] = rk1;
	rk[2] = rk2;
	rk[3] = rk3;
	rk[4] = rk4;
	rk[5] = rk5;
	rk[6] = rk6;
	rk[7] = rk7;

	for (u8 roundCount = 0; roundCount < ROUND_COUNT_256; roundCount++) {
		u32 temp = rk7;
		rk0 = rk0 ^ T4_3[(temp >> 16) & 0xff] ^ T4_2[(temp >> 8) & 0xff] ^ T4_1[(temp) & 0xff] ^ T4_0[(temp >> 24)] ^ RCON32[roundCount];
		rk1 = rk1 ^ rk0;
		rk2 = rk2 ^ rk1;
		rk3 = rk3 ^ rk2;
		rk4 = rk4 ^ T4_3[(rk3 >> 24) & 0xff] ^ T4_2[(rk3 >> 16) & 0xff] ^ T4_1[(rk3 >> 8) & 0xff] ^ T4_0[rk3 & 0xff];
		rk5 = rk5 ^ rk4;
		rk6 = rk6 ^ rk5;
		rk7 = rk7 ^ rk6;

		rk[roundCount * 8 + 8] = rk0;
		rk[roundCount * 8 + 9] = rk1;
		rk[roundCount * 8 + 10] = rk2;
		rk[roundCount * 8 + 11] = rk3;
		if (roundCount == 6) {
			break;
		}
		rk[roundCount * 8 + 12] = rk4;
		rk[roundCount * 8 + 13] = rk5;
		rk[roundCount * 8 + 14] = rk6;
		rk[roundCount * 8 + 15] = rk7;

	}

	//for (int i = 0; i < 60; i++) {
	//	printf("%08x ", rk[i]);
	//	if ((i + 1) % 4 == 0) {
	//		printf("Round: %d\n", i / 4);
	//	}
	//}
}

// CTR encryption with one table extended as 32 columns
// 1 Table [256][32] -> arithmetic shift: __byte_perm function
// SBox[256] is partly expanded
__global__ void counter256WithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBox(u32* pt, u32* rk, u32* t0G, u32* t4G, u64* range) {

	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int warpThreadIndex = threadIdx.x & 31;
	int warpThreadIndexSBox = warpThreadIndex % S_BOX_BANK_SIZE;

	// <SHARED MEMORY>
	__shared__ u32 t0S[TABLE_SIZE][SHARED_MEM_BANK_SIZE];
	__shared__ u32 t4S[TABLE_SIZE][S_BOX_BANK_SIZE];
	__shared__ u32 rkS[AES_256_KEY_SIZE_INT];

	if (threadIdx.x < TABLE_SIZE) {
		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) {
			t0S[threadIdx.x][bankIndex] = t0G[threadIdx.x];
		}

		for (u8 bankIndex = 0; bankIndex < S_BOX_BANK_SIZE; bankIndex++) {
			t4S[threadIdx.x][bankIndex] = t4G[threadIdx.x];
		}

		if (threadIdx.x < AES_256_KEY_SIZE_INT) {
			rkS[threadIdx.x] = rk[threadIdx.x];
		}

	}
	// </SHARED MEMORY>

	// Wait until every thread is ready
	__syncthreads();

	u32 pt0Init, pt1Init, pt2Init, pt3Init;
	u32 s0, s1, s2, s3;
	pt0Init = pt[0];
	pt1Init = pt[1];
	pt2Init = pt[2];
	pt3Init = pt[3];

	u32 threadRange = *range;
	u64 threadRangeStart = pt2Init;
	threadRangeStart = threadRangeStart << 32;
	threadRangeStart ^= pt3Init;
	threadRangeStart += (u64)threadIndex * threadRange;
	pt2Init = threadRangeStart >> 32;
	pt3Init = threadRangeStart & 0xFFFFFFFF;

	for (u32 rangeCount = 0; rangeCount < threadRange; rangeCount++) {

		// Create plaintext as 32 bit unsigned integers
		s0 = pt0Init;
		s1 = pt1Init;
		s2 = pt2Init;
		s3 = pt3Init;

		// First round just XORs input with key.
		s0 = s0 ^ rkS[0];
		s1 = s1 ^ rkS[1];
		s2 = s2 ^ rkS[2];
		s3 = s3 ^ rkS[3];

		u32 t0, t1, t2, t3;
		for (u8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1_256; roundCount++) {

			// Table based round function
			u32 rkStart = roundCount * 4 + 4;
			t0 = t0S[s0 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s3 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rkS[rkStart];
			t1 = t0S[s1 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s0 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rkS[rkStart + 1];
			t2 = t0S[s2 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s1 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rkS[rkStart + 2];
			t3 = t0S[s3 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s2 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rkS[rkStart + 3];

			s0 = t0;
			s1 = t1;
			s2 = t2;
			s3 = t3;

		}

		// Calculate the last round key
		// Last round uses s-box directly and XORs to produce output.
		s0 = (t4S[t0 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t1 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t2 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t3) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[56];
		s1 = (t4S[t1 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t2 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t3 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t0) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[57];
		s2 = (t4S[t2 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t3 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t0 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t1) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[58];
		s3 = (t4S[t3 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t0 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t1 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t2) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[59];

		//if (threadIndex == 0 && rangeCount == 0) {
		//printf("Ciphertext : %08x %08x %08x %08x\n", s0, s1, s2, s3);
		//}

		// Overflow
		if (pt3Init == MAX_U32) {
			pt2Init++;
		}

		// Create key as 32 bit unsigned integers
		pt3Init++;
	}

	if (threadIndex == 1048575) {
		printf("Plaintext : %08x %08x %08x %08x\n", pt0Init, pt1Init, pt2Init, pt3Init);
		printf("Ciphertext : %08x %08x %08x %08x\n", s0, s1, s2, s3);
		printf("-------------------------------\n");
	}

}

__host__ int main256Ctr() {
	printf("\n");
	printf("########## AES-256 Counter Mode Implementation ##########\n");
	printf("\n");

	// Allocate plaintext and every round key
	u32 *pt, *ct, *rk256, *roundKeys256;
	gpuErrorCheck(cudaMallocManaged(&pt, 4 * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&ct, 4 * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&rk256, 8 * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&roundKeys256, AES_256_KEY_SIZE_INT * sizeof(u32)));

	pt[0] = 0x6bc1bee2U;
	pt[1] = 0x2e409f96U;
	pt[2] = 0xe93d7e11U;
	pt[3] = 0x7393172aU;

	ct[0] = 0xF3EED1BDU;
	ct[1] = 0xB5D2A03CU;
	ct[2] = 0x064B5A7EU;
	ct[3] = 0x3DB181F8U;

	rk256[0] = 0x603deb10U;
	rk256[1] = 0x15ca71beU;
	rk256[2] = 0x2b73aef0U;
	rk256[3] = 0x857d7781U;
	rk256[4] = 0x1f352c07U;
	rk256[5] = 0x3b6108d7U;
	rk256[6] = 0x2d9810a3U;
	rk256[7] = 0x0914dff4U;

	// Allocate RCON values
	u32* rcon;
	gpuErrorCheck(cudaMallocManaged(&rcon, RCON_SIZE * sizeof(u32)));
	for (int i = 0; i < RCON_SIZE; i++) {
		rcon[i] = RCON32[i];
	}

	// Allocate Tables
	u32 *t0, *t1, *t2, *t3, *t4, *t4_0, *t4_1, *t4_2, *t4_3;
	gpuErrorCheck(cudaMallocManaged(&t0, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t1, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t2, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t3, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t4, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t4_0, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t4_1, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t4_2, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t4_3, TABLE_SIZE * sizeof(u32)));
	for (int i = 0; i < TABLE_SIZE; i++) {
		t0[i] = T0[i];
		t1[i] = T1[i];
		t2[i] = T2[i];
		t3[i] = T3[i];
		t4[i] = T4[i];
		t4_0[i] = T4_0[i];
		t4_1[i] = T4_1[i];
		t4_2[i] = T4_2[i];
		t4_3[i] = T4_3[i];
	}

	printf("-------------------------------\n");
	u64* range = calculateRange();
/*	printf("Initial Plaintext              : %08x %08x %08x %08x\n", pt[0], pt[1], pt[2], pt[3]);
	printf("Initial Key                    : %08x %08x %08x %08x %08x %08x %08x %08x\n", rk256[0], rk256[1], rk256[2], rk256[3], rk256[4], rk256[5], rk256[6], rk256[7]);
	printf("-------------------------------\n");*/

	keyExpansion256(rk256, roundKeys256);
	clock_t beginTime = clock();
	// Kernels
	counter256WithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBox<<<BLOCKS, THREADS>>>(pt, roundKeys256, t0, t4, range);

	cudaDeviceSynchronize();
	printf("Time elapsed: %f sec\n", float(clock() - beginTime) / CLOCKS_PER_SEC);
	printf("-------------------------------\n");
	printLastCUDAError();

	// Free alocated arrays
	cudaFree(range);
	cudaFree(pt);
	cudaFree(ct);
	cudaFree(rk256);
	cudaFree(roundKeys256);
	cudaFree(t0);
	cudaFree(t1);
	cudaFree(t2);
	cudaFree(t3);
	cudaFree(t4);
	cudaFree(t4_0);
	cudaFree(t4_1);
	cudaFree(t4_2);
	cudaFree(t4_3);
	cudaFree(rcon);
	

	return 0;
}