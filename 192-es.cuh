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


// Exhaustive search with one table extended as 32 columns
// 1 Table [256][32] -> arithmetic shift: __byte_perm function
// SBox[256] is partly expanded
__global__ void exhaustiveSearch192WithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBox(u32* pt, u32* ct, u32* rk, u32* t0G, u32* t4G, u32* rconG, u64* range) {

	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int warpThreadIndex = threadIdx.x & 31;
	int warpThreadIndexSBox = warpThreadIndex % S_BOX_BANK_SIZE;

	// <SHARED MEMORY>
	__shared__ u32 t0S[TABLE_SIZE][SHARED_MEM_BANK_SIZE];
	__shared__ u32 t4S[TABLE_SIZE][S_BOX_BANK_SIZE];
	__shared__ u32 rconS[RCON_SIZE];
	__shared__ u32 ctS[U32_SIZE];


	if (threadIdx.x < TABLE_SIZE) {
		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) {
			t0S[threadIdx.x][bankIndex] = t0G[threadIdx.x];
		}

		for (u8 bankIndex = 0; bankIndex < S_BOX_BANK_SIZE; bankIndex++) {
			t4S[threadIdx.x][bankIndex] = t4G[threadIdx.x];
		}

		if (threadIdx.x < RCON_SIZE) {
			rconS[threadIdx.x] = rconG[threadIdx.x];
		}

		if (threadIdx.x < U32_SIZE) {
			ctS[threadIdx.x] = ct[threadIdx.x];
		}
	}
	// </SHARED MEMORY>

	// Wait until every thread is ready
	__syncthreads();

	u32 rk0Init, rk1Init, rk2Init, rk3Init, rk4Init, rk5Init;
	rk0Init = rk[0];
	rk1Init = rk[1];
	rk2Init = rk[2];
	rk3Init = rk[3];
	rk4Init = rk[4];
	rk5Init = rk[5];

	u32 pt0Init, pt1Init, pt2Init, pt3Init;
	pt0Init = pt[0];
	pt1Init = pt[1];
	pt2Init = pt[2];
	pt3Init = pt[3];

	u32 threadRange = *range;
	u64 threadRangeStart = (u64)threadIndex * threadRange;
	rk4Init = rk4Init + threadRangeStart / MAX_U32;
	rk5Init = rk5Init + threadRangeStart % MAX_U32;

	for (u32 rangeCount = 0; rangeCount < threadRange; rangeCount++) {

		// Calculate round keys
		u32 rk0, rk1, rk2, rk3, rk4, rk5;
		rk0 = rk0Init;
		rk1 = rk1Init;
		rk2 = rk2Init;
		rk3 = rk3Init;
		rk4 = rk4Init;
		rk5 = rk5Init;

		// Create plaintext as 32 bit unsigned integers
		u32 s0, s1, s2, s3;
		s0 = pt0Init;
		s1 = pt1Init;
		s2 = pt2Init;
		s3 = pt3Init;

		// First round just XORs input with key.
		s0 = s0 ^ rk0;
		s1 = s1 ^ rk1;
		s2 = s2 ^ rk2;
		s3 = s3 ^ rk3;

		u32 t0, t1, t2, t3;
		u8 rconIndex = 0;
		for (u8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1_192; roundCount++) {
			// Table based round function
			t0 = t0S[s0 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s3 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT);
			t1 = t0S[s1 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s0 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT);
			t2 = t0S[s2 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s1 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT);
			t3 = t0S[s3 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s2 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT);

			// Add round key
			if (roundCount % 3 == 0) {
				t0 = t0 ^ rk4;
				t1 = t1 ^ rk5;
				// Calculate round key
				u32 temp = rk5;
				rk0 = rk0 ^
					(t4S[(temp >> 16) & 0xff][warpThreadIndexSBox] & 0xff000000) ^
					(t4S[(temp >> 8) & 0xff][warpThreadIndexSBox] & 0x00ff0000) ^
					(t4S[(temp) & 0xff][warpThreadIndexSBox] & 0x0000ff00) ^
					(t4S[(temp >> 24)][warpThreadIndexSBox] & 0x000000ff) ^
					rconS[rconIndex++];
				rk1 = rk1 ^ rk0;
				rk2 = rk2 ^ rk1;
				rk3 = rk3 ^ rk2;
				rk4 = rk4 ^ rk3;
				rk5 = rk5 ^ rk4;

				t2 = t2 ^ rk0;
				t3 = t3 ^ rk1;
			} else if (roundCount % 3 == 1) {
				t0 = t0 ^ rk2;
				t1 = t1 ^ rk3;
				t2 = t2 ^ rk4;
				t3 = t3 ^ rk5;
			} else {
				// Calculate round key
				u32 temp = rk5;
				rk0 = rk0 ^
					(t4S[(temp >> 16) & 0xff][warpThreadIndexSBox] & 0xff000000) ^
					(t4S[(temp >> 8) & 0xff][warpThreadIndexSBox] & 0x00ff0000) ^
					(t4S[(temp) & 0xff][warpThreadIndexSBox] & 0x0000ff00) ^
					(t4S[(temp >> 24)][warpThreadIndexSBox] & 0x000000ff) ^
					rconS[rconIndex++];
				rk1 = rk1 ^ rk0;
				rk2 = rk2 ^ rk1;
				rk3 = rk3 ^ rk2;
				rk4 = rk4 ^ rk3;
				rk5 = rk5 ^ rk4;

				t0 = t0 ^ rk0;
				t1 = t1 ^ rk1;
				t2 = t2 ^ rk2;
				t3 = t3 ^ rk3;
			}

			s0 = t0;
			s1 = t1;
			s2 = t2;
			s3 = t3;
		}

		// Calculate the last round key
		u32 temp = rk5;
		rk0 = rk0 ^
			(t4S[(temp >> 16) & 0xff][warpThreadIndexSBox] & 0xff000000) ^
			(t4S[(temp >> 8) & 0xff][warpThreadIndexSBox] & 0x00ff0000) ^
			(t4S[(temp) & 0xff][warpThreadIndexSBox] & 0x0000ff00) ^
			(t4S[(temp >> 24)][warpThreadIndexSBox] & 0x000000ff) ^
			rconS[rconIndex];

		// Last round uses s-box directly and XORs to produce output.
		s0 = (t4S[t0 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t1 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t2 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t3) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rk0;
		if (s0 == ctS[0]) {
			rk1 = rk1 ^ rk0;
			s1 = (t4S[t1 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t2 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t3 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t0) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rk1;
			if (s1 == ctS[1]) {
				rk2 = rk2 ^ rk1;
				s2 = (t4S[t2 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t3 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t0 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t1) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rk2;
				if (s2 == ctS[2]) {
					rk3 = rk2 ^ rk3;
					s3 = (t4S[t3 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t0 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t1 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t2) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rk3;
					if (s3 == ctS[3]) {
						printf("! Found key : %08x %08x %08x %08x %08x %08x\n", rk0Init, rk1Init, rk2Init, rk3Init, rk4Init, rk5Init);
						printf("-------------------------------\n");
					}
				}
			}
		}

		// Overflow
		if (rk5Init == MAX_U32) {
			rk4Init++;
		}

		// Create key as 32 bit unsigned integers
		rk5Init++;
	}
}
__global__ void exhaustiveSearch192WithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBoxCihangir(u32* pt, u32* ct, u32* rk, u32* t0G, u32* t4G, u32* rconG, u64* range, u8* SAES) {
	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int warpThreadIndex = threadIdx.x & 31;
//	int warpThreadIndexSBox = warpThreadIndex % S_BOX_BANK_SIZE;
	// <SHARED MEMORY>
	__shared__ u32 t0S[TABLE_SIZE][SHARED_MEM_BANK_SIZE];
//	__shared__ u32 t4S[TABLE_SIZE][S_BOX_BANK_SIZE];
	__shared__ u8 Sbox[64][32][4];
	__shared__ u32 rconS[RCON_SIZE];
	__shared__ u32 ctS[U32_SIZE];
	if (threadIdx.x < TABLE_SIZE) {
		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) {			t0S[threadIdx.x][bankIndex] = t0G[threadIdx.x];		}
//		for (u8 bankIndex = 0; bankIndex < S_BOX_BANK_SIZE; bankIndex++) {			t4S[threadIdx.x][bankIndex] = t4G[threadIdx.x];		}
		for (u8 bankIndex = 0; bankIndex < S_BOX_BANK_SIZE; bankIndex++) { Sbox[threadIdx.x / 4][bankIndex][threadIdx.x % 4] = SAES[threadIdx.x]; }
		if (threadIdx.x < RCON_SIZE) {			rconS[threadIdx.x] = rconG[threadIdx.x];		}
		if (threadIdx.x < U32_SIZE) {			ctS[threadIdx.x] = ct[threadIdx.x];		}
	}
	// </SHARED MEMORY>
	// Wait until every thread is ready
	__syncthreads();
	u32 rk0Init, rk1Init, rk2Init, rk3Init, rk4Init, rk5Init;
	rk0Init = rk[0];	rk1Init = rk[1];	rk2Init = rk[2];	rk3Init = rk[3];	rk4Init = rk[4];	rk5Init = rk[5];
	u32 pt0Init, pt1Init, pt2Init, pt3Init;
	pt0Init = pt[0];	pt1Init = pt[1];	pt2Init = pt[2];	pt3Init = pt[3];
	u32 threadRange = *range;
	u64 threadRangeStart = (u64)threadIndex * threadRange;
	rk4Init = rk4Init + threadRangeStart / MAX_U32;
	rk5Init = rk5Init + threadRangeStart % MAX_U32;
	for (u32 rangeCount = 0; rangeCount < threadRange; rangeCount++) {
		// Calculate round keys
		u32 rk0, rk1, rk2, rk3, rk4, rk5;
		rk0 = rk0Init;		rk1 = rk1Init;		rk2 = rk2Init;		rk3 = rk3Init;		rk4 = rk4Init;		rk5 = rk5Init;
		// Create plaintext as 32 bit unsigned integers
		u32 s0, s1, s2, s3;
		s0 = pt0Init;		s1 = pt1Init;		s2 = pt2Init;		s3 = pt3Init;
		// First round just XORs input with key.
		s0 = s0 ^ rk0;		s1 = s1 ^ rk1;		s2 = s2 ^ rk2;		s3 = s3 ^ rk3;
		u32 t0, t1, t2, t3;
		u8 rconIndex = 0;
		for (u8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1_192; roundCount++) {
			// Table based round function
			t0 = t0S[s0 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s3 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT);
			t1 = t0S[s1 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s0 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT);
			t2 = t0S[s2 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s1 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT);
			t3 = t0S[s3 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s2 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT);
			// Add round key
			if (roundCount % 3 == 0) {
				t0 = t0 ^ rk4;				t1 = t1 ^ rk5;
				// Calculate round key
				u32 temp = rk5;
				rk0 = rk0 ^
					arithmeticRightShiftBytePerm((u64)Sbox[((temp >> 16) & 0xff) / 4][warpThreadIndex][((temp >> 16) & 0xff) % 4], SHIFT_1_RIGHT) ^
					arithmeticRightShiftBytePerm((u64)Sbox[((temp >> 8) & 0xff) / 4][warpThreadIndex][((temp >> 8) & 0xff) % 4], SHIFT_2_RIGHT) ^
					arithmeticRightShiftBytePerm((u64)Sbox[((temp) & 0xff) / 4][warpThreadIndex][((temp) & 0xff) % 4], SHIFT_3_RIGHT) ^
					((u64)Sbox[((temp >> 24) / 4)][warpThreadIndex][((temp >> 24) % 4)]) ^
					rconS[rconIndex++];
				rk1 = rk1 ^ rk0;				rk2 = rk2 ^ rk1;				rk3 = rk3 ^ rk2;				rk4 = rk4 ^ rk3;				rk5 = rk5 ^ rk4;
				t2 = t2 ^ rk0;
				t3 = t3 ^ rk1;
			}
			else if (roundCount % 3 == 1) {				t0 = t0 ^ rk2;				t1 = t1 ^ rk3;				t2 = t2 ^ rk4;				t3 = t3 ^ rk5;			}
			else {
				// Calculate round key
				u32 temp = rk5;
				rk0 = rk0 ^
					arithmeticRightShiftBytePerm((u64)Sbox[((temp >> 16) & 0xff) / 4][warpThreadIndex][((temp >> 16) & 0xff) % 4], SHIFT_1_RIGHT) ^
					arithmeticRightShiftBytePerm((u64)Sbox[((temp >> 8) & 0xff) / 4][warpThreadIndex][((temp >> 8) & 0xff) % 4], SHIFT_2_RIGHT) ^
					arithmeticRightShiftBytePerm((u64)Sbox[((temp) & 0xff) / 4][warpThreadIndex][((temp) & 0xff) % 4], SHIFT_3_RIGHT) ^
					((u64)Sbox[((temp >> 24) / 4)][warpThreadIndex][((temp >> 24) % 4)]) ^
					rconS[rconIndex++];
				rk1 = rk1 ^ rk0;				rk2 = rk2 ^ rk1;				rk3 = rk3 ^ rk2;				rk4 = rk4 ^ rk3;				rk5 = rk5 ^ rk4;
				t0 = t0 ^ rk0;				t1 = t1 ^ rk1;				t2 = t2 ^ rk2;				t3 = t3 ^ rk3;
			}
			s0 = t0;			s1 = t1;			s2 = t2;			s3 = t3;
		}
		// Calculate the last round key
		u32 temp = rk5;
		rk0 = rk0 ^
			arithmeticRightShiftBytePerm((u64)Sbox[((temp >> 16) & 0xff) / 4][warpThreadIndex][((temp >> 16) & 0xff) % 4], SHIFT_1_RIGHT) ^
			arithmeticRightShiftBytePerm((u64)Sbox[((temp >> 8) & 0xff) / 4][warpThreadIndex][((temp >> 8) & 0xff) % 4], SHIFT_2_RIGHT) ^
			arithmeticRightShiftBytePerm((u64)Sbox[((temp) & 0xff) / 4][warpThreadIndex][((temp) & 0xff) % 4], SHIFT_3_RIGHT) ^
			((u64)Sbox[((temp >> 24) / 4)][warpThreadIndex][((temp >> 24) % 4)]) ^
			rconS[rconIndex];

		// Last round uses s-box directly and XORs to produce output.
		s0 = arithmeticRightShiftBytePerm((u64)Sbox[((t0 >> 24)) / 4][warpThreadIndex][((t0 >> 24)) % 4], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm((u64)Sbox[((t1 >> 16) & 0xff) / 4][warpThreadIndex][((t1 >> 16)) % 4], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm((u64)Sbox[((t2 >> 8) & 0xFF) / 4][warpThreadIndex][((t2 >> 8)) % 4], SHIFT_3_RIGHT) ^ ((u64)Sbox[((t3 & 0xFF) / 4)][warpThreadIndex][((t3 & 0xFF) % 4)]) ^ rk0;
		if (s0 == ctS[0]) {
			rk1 = rk1 ^ rk0;
			s1 = arithmeticRightShiftBytePerm((u64)Sbox[((t1 >> 24)) / 4][warpThreadIndex][((t1 >> 24)) % 4], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm((u64)Sbox[((t2 >> 16) & 0xff) / 4][warpThreadIndex][((t2 >> 16)) % 4], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm((u64)Sbox[((t3 >> 8) & 0xFF) / 4][warpThreadIndex][((t3 >> 8)) % 4], SHIFT_3_RIGHT) ^ ((u64)Sbox[((t0 & 0xFF) / 4)][warpThreadIndex][((t0 & 0xFF) % 4)]) ^ rk1;
			if (s1 == ctS[1]) {
				rk2 = rk2 ^ rk1;
				s2 = arithmeticRightShiftBytePerm((u64)Sbox[((t2 >> 24)) / 4][warpThreadIndex][((t2 >> 24)) % 4], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm((u64)Sbox[((t3 >> 16) & 0xff) / 4][warpThreadIndex][((t3 >> 16)) % 4], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm((u64)Sbox[((t0 >> 8) & 0xFF) / 4][warpThreadIndex][((t0 >> 8)) % 4], SHIFT_3_RIGHT) ^ ((u64)Sbox[((t1 & 0xFF) / 4)][warpThreadIndex][((t1 & 0xFF) % 4)]) ^ rk2;
				if (s2 == ctS[2]) {
					rk3 = rk2 ^ rk3;
					s3 = arithmeticRightShiftBytePerm((u64)Sbox[((t3 >> 24)) / 4][warpThreadIndex][((t3 >> 24)) % 4], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm((u64)Sbox[((t0 >> 16) & 0xff) / 4][warpThreadIndex][((t0 >> 16)) % 4], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm((u64)Sbox[((t1 >> 8) & 0xFF) / 4][warpThreadIndex][((t1 >> 8)) % 4], SHIFT_3_RIGHT) ^ ((u64)Sbox[((t2 & 0xFF) / 4)][warpThreadIndex][((t2 & 0xFF) % 4)]) ^ rk3;
					if (s3 == ctS[3]) {
						printf("! Found key : %08x %08x %08x %08x %08x %08x\n", rk0Init, rk1Init, rk2Init, rk3Init, rk4Init, rk5Init);
						printf("-------------------------------\n");
					}
				}
			}
		}
		if (rk5Init == MAX_U32) { rk4Init++; }// Overflow
		rk5Init++; // Create key as 32 bit unsigned integers
	}
}

__host__ int main192ExhaustiveSearch() {
	printf("\n");	printf("########## AES-192 Exhaustive Search Implementation ##########\n");	printf("\n");
	// Allocate plaintext, ciphertext and initial round key
	u32 *pt, *ct, *rk192;
	gpuErrorCheck(cudaMallocManaged(&pt, 4 * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&ct, 4 * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&rk192, 6 * sizeof(u32)));
	pt[0] = 0x6bc1bee2U;	pt[1] = 0x2e409f96U;	pt[2] = 0xe93d7e11U;	pt[3] = 0x7393172aU;
	ct[0] = 0xBD334F1DU;	ct[1] = 0x6E45F25FU;	ct[2] = 0xF712A214U;	ct[3] = 0x571FA5CCU;
	rk192[0] = 0x8e73b0f7U;	rk192[1] = 0xda0e6452U;	rk192[2] = 0xc810f32bU;	rk192[3] = 0x809079e5U;	rk192[4] = 0x62f8ead2U;	rk192[5] = 0x522c6b70U;
	// Allocate RCON values
	u32* rcon;
	gpuErrorCheck(cudaMallocManaged(&rcon, RCON_SIZE * sizeof(u32)));
	for (int i = 0; i < RCON_SIZE; i++) {		rcon[i] = RCON32[i];	}
	// Allocate Tables
	u32 *t0, *t1, *t2, *t3, *t4, *t4_0, *t4_1, *t4_2, *t4_3;
	u8* SAES_d; // Cihangir
	gpuErrorCheck(cudaMallocManaged(&t0, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t1, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t2, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t3, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t4, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t4_0, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t4_1, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t4_2, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t4_3, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&SAES_d, 256 * sizeof(u8))); // Cihangir
	for (int i = 0; i < TABLE_SIZE; i++) {		t0[i] = T0[i]; t1[i] = T1[i]; t2[i] = T2[i]; t3[i] = T3[i]; t4[i] = T4[i]; t4_0[i] = T4_0[i]; t4_1[i] = T4_1[i]; t4_2[i] = T4_2[i]; t4_3[i] = T4_3[i];	}
	for (int i = 0; i < 256; i++) SAES_d[i] = SAES[i];
	printf("-------------------------------\n");
	u64* range = calculateRange();
/*	printf("Plaintext                      : %08x %08x %08x %08x\n", pt[0], pt[1], pt[2], pt[3]);
	printf("Ciphertext                     : %08x %08x %08x %08x\n", ct[0], ct[1], ct[2], ct[3]);
	printf("Initial Key                    : %08x %08x %08x %08x %08x %08x\n", rk192[0], rk192[1], rk192[2], rk192[3], rk192[4], rk192[5]);
	printf("-------------------------------\n");*/
	clock_t beginTime = clock();
	// Kernels
	exhaustiveSearch192WithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBox<<<BLOCKS, THREADS>>>(pt, ct, rk192, t0, t4, rcon, range);
//	exhaustiveSearch192WithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBoxCihangir << <BLOCKS, THREADS >> > (pt, ct, rk192, t0, t4, rcon, range, SAES_d);

	cudaDeviceSynchronize();
	printf("Time elapsed: %f sec\n", float(clock() - beginTime) / CLOCKS_PER_SEC);
	printf("-------------------------------\n");
	printLastCUDAError();
	// Free alocated arrays
	cudaFree(range);cudaFree(pt);cudaFree(ct);cudaFree(rk192);cudaFree(t0);cudaFree(t1);cudaFree(t2);cudaFree(t3);cudaFree(t4);
	cudaFree(t4_0);cudaFree(t4_1);cudaFree(t4_2);cudaFree(t4_3);cudaFree(rcon);
	return 0;
}