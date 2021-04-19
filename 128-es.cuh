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


// Basic exhaustive search
// 4 Tables
__global__ void exhaustiveSearch(u32* pt, u32* ct, u32* rk, u32* t0G, u32* t1G, u32* t2G, u32* t3G, u32* t4G, u32* rconG, u64* range) {

	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

	// <SHARED MEMORY>
	__shared__ u32 t0S[TABLE_SIZE];
	__shared__ u32 t1S[TABLE_SIZE];
	__shared__ u32 t2S[TABLE_SIZE];
	__shared__ u32 t3S[TABLE_SIZE];
	__shared__ u32 t4S[TABLE_SIZE];
	__shared__ u32 rconS[RCON_SIZE];
	__shared__ u32 ctS[U32_SIZE];


	if (threadIdx.x < TABLE_SIZE) {
		t0S[threadIdx.x] = t0G[threadIdx.x];
		t1S[threadIdx.x] = t1G[threadIdx.x];
		t2S[threadIdx.x] = t2G[threadIdx.x];
		t3S[threadIdx.x] = t3G[threadIdx.x];
		t4S[threadIdx.x] = t4G[threadIdx.x];

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

	u32 rk0Init, rk1Init, rk2Init, rk3Init;
	rk0Init = rk[0];
	rk1Init = rk[1];
	rk2Init = rk[2];
	rk3Init = rk[3];

	u32 pt0Init, pt1Init, pt2Init, pt3Init;
	pt0Init = pt[0];
	pt1Init = pt[1];
	pt2Init = pt[2];
	pt3Init = pt[3];

	u64 threadRange = *range;
	u64 threadRangeStart = (u64)threadIndex * threadRange;
	rk2Init = rk2Init + threadRangeStart / MAX_U32;
	rk3Init = rk3Init + threadRangeStart % MAX_U32;

	for (u64 rangeCount = 0; rangeCount < threadRange; rangeCount++) {

		u32 rk0, rk1, rk2, rk3;
		rk0 = rk0Init;
		rk1 = rk1Init;
		rk2 = rk2Init;
		rk3 = rk3Init;

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

		//if (threadIndex == 0 && rangeCount == 0) {
		//	printf("--Round: %d\n", 0);
		//	printf("%08x%08x%08x%08x\n", s0, s1, s2, s3);
		//	printf("-- Round Key\n");
		//	printf("%08x%08x%08x%08x\n", rk0, rk1, rk2, rk3);
		//}

		u32 t0, t1, t2, t3;
		for (u8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1; roundCount++) {

			// Calculate round key
			u32 temp = rk3;
			// TODO: temp & 0xff000000
			rk0 = rk0 ^
				(t4S[(temp >> 16) & 0xff] & 0xff000000) ^
				(t4S[(temp >> 8) & 0xff] & 0x00ff0000) ^
				(t4S[(temp) & 0xff] & 0x0000ff00) ^
				(t4S[(temp >> 24)] & 0x000000ff) ^
				rconS[roundCount];
			rk1 = rk1 ^ rk0;
			rk2 = rk2 ^ rk1;
			rk3 = rk2 ^ rk3;

			// Table based round function
			t0 = t0S[s0 >> 24] ^ t1S[(s1 >> 16) & 0xFF] ^ t2S[(s2 >> 8) & 0xFF] ^ t3S[s3 & 0xFF] ^ rk0;
			t1 = t0S[s1 >> 24] ^ t1S[(s2 >> 16) & 0xFF] ^ t2S[(s3 >> 8) & 0xFF] ^ t3S[s0 & 0xFF] ^ rk1;
			t2 = t0S[s2 >> 24] ^ t1S[(s3 >> 16) & 0xFF] ^ t2S[(s0 >> 8) & 0xFF] ^ t3S[s1 & 0xFF] ^ rk2;
			t3 = t0S[s3 >> 24] ^ t1S[(s0 >> 16) & 0xFF] ^ t2S[(s1 >> 8) & 0xFF] ^ t3S[s2 & 0xFF] ^ rk3;

			s0 = t0;
			s1 = t1;
			s2 = t2;
			s3 = t3;

			//if (threadIndex == 0 && rangeCount == 0) {
			//	printf("--Round: %d\n", roundCount);
			//	printf("%08x%08x%08x%08x\n", s0, s1, s2, s3);
			//	printf("-- Round Key\n");
			//	printf("%08x%08x%08x%08x\n", rk0, rk1, rk2, rk3);
			//}
		}

		// Calculate the last round key
		u32 temp = rk3;
		rk0 = rk0 ^
			(t4S[(temp >> 16) & 0xff] & 0xff000000) ^
			(t4S[(temp >> 8) & 0xff] & 0x00ff0000) ^
			(t4S[(temp) & 0xff] & 0x0000ff00) ^
			(t4S[(temp >> 24)] & 0x000000ff) ^
			rconS[ROUND_COUNT_MIN_1];
		// Last round uses s-box directly and XORs to produce output.
		s0 = (t4S[t0 >> 24] & 0xFF000000) ^ (t4S[(t1 >> 16) & 0xff] & 0x00FF0000) ^ (t4S[(t2 >> 8) & 0xff] & 0x0000FF00) ^ (t4S[(t3) & 0xFF] & 0x000000FF) ^ rk0;
		if (s0 == ctS[0]) {
			rk1 = rk1 ^ rk0;
			s1 = (t4S[t1 >> 24] & 0xFF000000) ^ (t4S[(t2 >> 16) & 0xff] & 0x00FF0000) ^ (t4S[(t3 >> 8) & 0xff] & 0x0000FF00) ^ (t4S[(t0) & 0xFF] & 0x000000FF) ^ rk1;
			if (s1 == ctS[1]) {
				rk2 = rk2 ^ rk1;
				s2 = (t4S[t2 >> 24] & 0xFF000000) ^ (t4S[(t3 >> 16) & 0xff] & 0x00FF0000) ^ (t4S[(t0 >> 8) & 0xff] & 0x0000FF00) ^ (t4S[(t1) & 0xFF] & 0x000000FF) ^ rk2;
				if (s2 == ctS[2]) {
					rk3 = rk2 ^ rk3;
					s3 = (t4S[t3 >> 24] & 0xFF000000) ^ (t4S[(t0 >> 16) & 0xff] & 0x00FF0000) ^ (t4S[(t1 >> 8) & 0xff] & 0x0000FF00) ^ (t4S[(t2) & 0xFF] & 0x000000FF) ^ rk3;
					if (s3 == ctS[3]) {
						printf("! Found key : %08x %08x %08x %08x\n", rk0Init, rk1Init, rk2Init, rk3Init);
						printf("-------------------------------\n");
					}
				}
			}
		}

		// Overflow
		if (rk3Init == MAX_U32) {
			rk2Init++;
		}

		// Create key as 32 bit unsigned integers
		rk3Init++;
	}
}

// Exhaustive search with one table
// 1 Table -> arithmetic shift: 2 shift 1 and
__global__ void exhaustiveSearchWithOneTable(u32* pt, u32* ct, u32* rk, u32* t0G, u32* t4G, u32* rconG, u64* range) {

	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

	// <SHARED MEMORY>
	__shared__ u32 t0S[TABLE_SIZE];
	__shared__ u32 t4S[TABLE_SIZE];
	__shared__ u32 rconS[RCON_SIZE];
	__shared__ u32 ctS[U32_SIZE];

	if (threadIdx.x < TABLE_SIZE) {
		t0S[threadIdx.x] = t0G[threadIdx.x];
		t4S[threadIdx.x] = t4G[threadIdx.x];

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

	u32 rk0Init, rk1Init, rk2Init, rk3Init;
	rk0Init = rk[0];
	rk1Init = rk[1];
	rk2Init = rk[2];
	rk3Init = rk[3];

	u32 pt0Init, pt1Init, pt2Init, pt3Init;
	pt0Init = pt[0];
	pt1Init = pt[1];
	pt2Init = pt[2];
	pt3Init = pt[3];

	u64 threadRange = *range;
	u64 threadRangeStart = (u64)threadIndex * threadRange;
	rk2Init = rk2Init + threadRangeStart / MAX_U32;
	rk3Init = rk3Init + threadRangeStart % MAX_U32;

	for (u64 rangeCount = 0; rangeCount < threadRange; rangeCount++) {

		u32 rk0, rk1, rk2, rk3;
		rk0 = rk0Init;
		rk1 = rk1Init;
		rk2 = rk2Init;
		rk3 = rk3Init;

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
		for (u8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1; roundCount++) {

			// Calculate round key
			u32 temp = rk3;
			rk0 = rk0 ^
				(t4S[(temp >> 16) & 0xff] & 0xff000000) ^
				(t4S[(temp >> 8) & 0xff] & 0x00ff0000) ^
				(t4S[(temp) & 0xff] & 0x0000ff00) ^
				(t4S[(temp >> 24)] & 0x000000ff) ^
				rconS[roundCount];
			rk1 = rk1 ^ rk0;
			rk2 = rk2 ^ rk1;
			rk3 = rk2 ^ rk3;

			// Table based round function
			t0 = t0S[s0 >> 24] ^ arithmeticRightShift(t0S[(s1 >> 16) & 0xFF], 8) ^ arithmeticRightShift(t0S[(s2 >> 8) & 0xFF], 16) ^ arithmeticRightShift(t0S[s3 & 0xFF], 24) ^ rk0;
			t1 = t0S[s1 >> 24] ^ arithmeticRightShift(t0S[(s2 >> 16) & 0xFF], 8) ^ arithmeticRightShift(t0S[(s3 >> 8) & 0xFF], 16) ^ arithmeticRightShift(t0S[s0 & 0xFF], 24) ^ rk1;
			t2 = t0S[s2 >> 24] ^ arithmeticRightShift(t0S[(s3 >> 16) & 0xFF], 8) ^ arithmeticRightShift(t0S[(s0 >> 8) & 0xFF], 16) ^ arithmeticRightShift(t0S[s1 & 0xFF], 24) ^ rk2;
			t3 = t0S[s3 >> 24] ^ arithmeticRightShift(t0S[(s0 >> 16) & 0xFF], 8) ^ arithmeticRightShift(t0S[(s1 >> 8) & 0xFF], 16) ^ arithmeticRightShift(t0S[s2 & 0xFF], 24) ^ rk3;

			s0 = t0;
			s1 = t1;
			s2 = t2;
			s3 = t3;

		}

		// Calculate the last round key
		u32 temp = rk3;
		rk0 = rk0 ^
			(t4S[(temp >> 16) & 0xff] & 0xff000000) ^
			(t4S[(temp >> 8) & 0xff] & 0x00ff0000) ^
			(t4S[(temp) & 0xff] & 0x0000ff00) ^
			(t4S[(temp >> 24)] & 0x000000ff) ^
			rconS[ROUND_COUNT_MIN_1];
		// Last round uses s-box directly and XORs to produce output.
		s0 = (t4S[t0 >> 24] & 0xFF000000) ^ (t4S[(t1 >> 16) & 0xff] & 0x00FF0000) ^ (t4S[(t2 >> 8) & 0xff] & 0x0000FF00) ^ (t4S[(t3) & 0xFF] & 0x000000FF) ^ rk0;
		if (s0 == ctS[0]) {
			rk1 = rk1 ^ rk0;
			s1 = (t4S[t1 >> 24] & 0xFF000000) ^ (t4S[(t2 >> 16) & 0xff] & 0x00FF0000) ^ (t4S[(t3 >> 8) & 0xff] & 0x0000FF00) ^ (t4S[(t0) & 0xFF] & 0x000000FF) ^ rk1;
			if (s1 == ctS[1]) {
				rk2 = rk2 ^ rk1;
				s2 = (t4S[t2 >> 24] & 0xFF000000) ^ (t4S[(t3 >> 16) & 0xff] & 0x00FF0000) ^ (t4S[(t0 >> 8) & 0xff] & 0x0000FF00) ^ (t4S[(t1) & 0xFF] & 0x000000FF) ^ rk2;
				if (s2 == ctS[2]) {
					rk3 = rk2 ^ rk3;
					s3 = (t4S[t3 >> 24] & 0xFF000000) ^ (t4S[(t0 >> 16) & 0xff] & 0x00FF0000) ^ (t4S[(t1 >> 8) & 0xff] & 0x0000FF00) ^ (t4S[(t2) & 0xFF] & 0x000000FF) ^ rk3;
					if (s3 == ctS[3]) {
						printf("! Found key : %08x %08x %08x %08x\n", rk0Init, rk1Init, rk2Init, rk3Init);
						printf("-------------------------------\n");
					}
				}
			}
		}

		// Overflow
		if (rk3Init == MAX_U32) {
			rk2Init++;
		}

		// Create key as 32 bit unsigned integers
		rk3Init++;
	}
}

// Exhaustive search with one table extended as 32 columns
// 1 Table [256][32] -> arithmetic shift: 2 shift 1 and
__global__ void exhaustiveSearchWithOneTableExtendedSharedMemory(u32* pt, u32* ct, u32* rk, u32* t0G, u32* t4G, u32* rconG, u64* range) {

	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int warpThreadIndex = threadIdx.x & 31;

	// <SHARED MEMORY>
	__shared__ u32 t0S[TABLE_SIZE][SHARED_MEM_BANK_SIZE];
	__shared__ u32 t4S[TABLE_SIZE];
	__shared__ u32 rconS[RCON_SIZE];
	__shared__ u32 ctS[U32_SIZE];

	if (threadIdx.x < TABLE_SIZE) {
		t4S[threadIdx.x] = t4G[threadIdx.x];
		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) {
			t0S[threadIdx.x][bankIndex] = t0G[threadIdx.x];
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

	u32 rk0Init, rk1Init, rk2Init, rk3Init;
	rk0Init = rk[0];
	rk1Init = rk[1];
	rk2Init = rk[2];
	rk3Init = rk[3];

	u32 pt0Init, pt1Init, pt2Init, pt3Init;
	pt0Init = pt[0];
	pt1Init = pt[1];
	pt2Init = pt[2];
	pt3Init = pt[3];

	u64 threadRange = *range;
	u64 threadRangeStart = (u64)threadIndex * threadRange;
	rk2Init = rk2Init + threadRangeStart / MAX_U32;
	rk3Init = rk3Init + threadRangeStart % MAX_U32;

	for (u64 rangeCount = 0; rangeCount < threadRange; rangeCount++) {

		u32 rk0, rk1, rk2, rk3;
		rk0 = rk0Init;
		rk1 = rk1Init;
		rk2 = rk2Init;
		rk3 = rk3Init;

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
		for (u8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1; roundCount++) {

			// Calculate round key
			u32 temp = rk3;
			rk0 = rk0 ^
				(t4S[(temp >> 16) & 0xff] & 0xff000000) ^
				(t4S[(temp >> 8) & 0xff] & 0x00ff0000) ^
				(t4S[(temp) & 0xff] & 0x0000ff00) ^
				(t4S[(temp >> 24)] & 0x000000ff) ^
				rconS[roundCount];
			rk1 = rk1 ^ rk0;
			rk2 = rk2 ^ rk1;
			rk3 = rk2 ^ rk3;

			// Table based round function
			t0 = t0S[s0 >> 24][warpThreadIndex] ^ arithmeticRightShift(t0S[(s1 >> 16) & 0xFF][warpThreadIndex], 8) ^ arithmeticRightShift(t0S[(s2 >> 8) & 0xFF][warpThreadIndex], 16) ^ arithmeticRightShift(t0S[s3 & 0xFF][warpThreadIndex], 24) ^ rk0;
			t1 = t0S[s1 >> 24][warpThreadIndex] ^ arithmeticRightShift(t0S[(s2 >> 16) & 0xFF][warpThreadIndex], 8) ^ arithmeticRightShift(t0S[(s3 >> 8) & 0xFF][warpThreadIndex], 16) ^ arithmeticRightShift(t0S[s0 & 0xFF][warpThreadIndex], 24) ^ rk1;
			t2 = t0S[s2 >> 24][warpThreadIndex] ^ arithmeticRightShift(t0S[(s3 >> 16) & 0xFF][warpThreadIndex], 8) ^ arithmeticRightShift(t0S[(s0 >> 8) & 0xFF][warpThreadIndex], 16) ^ arithmeticRightShift(t0S[s1 & 0xFF][warpThreadIndex], 24) ^ rk2;
			t3 = t0S[s3 >> 24][warpThreadIndex] ^ arithmeticRightShift(t0S[(s0 >> 16) & 0xFF][warpThreadIndex], 8) ^ arithmeticRightShift(t0S[(s1 >> 8) & 0xFF][warpThreadIndex], 16) ^ arithmeticRightShift(t0S[s2 & 0xFF][warpThreadIndex], 24) ^ rk3;

			s0 = t0;
			s1 = t1;
			s2 = t2;
			s3 = t3;

		}

		// Calculate the last round key
		u32 temp = rk3;
		rk0 = rk0 ^
			(t4S[(temp >> 16) & 0xff] & 0xff000000) ^
			(t4S[(temp >> 8) & 0xff] & 0x00ff0000) ^
			(t4S[(temp) & 0xff] & 0x0000ff00) ^
			(t4S[(temp >> 24)] & 0x000000ff) ^
			rconS[ROUND_COUNT_MIN_1];
		// Last round uses s-box directly and XORs to produce output.
		s0 = (t4S[t0 >> 24] & 0xFF000000) ^ (t4S[(t1 >> 16) & 0xff] & 0x00FF0000) ^ (t4S[(t2 >> 8) & 0xff] & 0x0000FF00) ^ (t4S[(t3) & 0xFF] & 0x000000FF) ^ rk0;
		if (s0 == ctS[0]) {
			rk1 = rk1 ^ rk0;
			s1 = (t4S[t1 >> 24] & 0xFF000000) ^ (t4S[(t2 >> 16) & 0xff] & 0x00FF0000) ^ (t4S[(t3 >> 8) & 0xff] & 0x0000FF00) ^ (t4S[(t0) & 0xFF] & 0x000000FF) ^ rk1;
			if (s1 == ctS[1]) {
				rk2 = rk2 ^ rk1;
				s2 = (t4S[t2 >> 24] & 0xFF000000) ^ (t4S[(t3 >> 16) & 0xff] & 0x00FF0000) ^ (t4S[(t0 >> 8) & 0xff] & 0x0000FF00) ^ (t4S[(t1) & 0xFF] & 0x000000FF) ^ rk2;
				if (s2 == ctS[2]) {
					rk3 = rk2 ^ rk3;
					s3 = (t4S[t3 >> 24] & 0xFF000000) ^ (t4S[(t0 >> 16) & 0xff] & 0x00FF0000) ^ (t4S[(t1 >> 8) & 0xff] & 0x0000FF00) ^ (t4S[(t2) & 0xFF] & 0x000000FF) ^ rk3;
					if (s3 == ctS[3]) {
						printf("! Found key : %08x %08x %08x %08x\n", rk0Init, rk1Init, rk2Init, rk3Init);
						printf("-------------------------------\n");
					}
				}
			}
		}

		// Overflow
		if (rk3Init == MAX_U32) {
			rk2Init++;
		}

		// Create key as 32 bit unsigned integers
		rk3Init++;
	}
}

// Exhaustive search with one table extended as 32 columns
// 1 Table [256][32] -> arithmetic shift: __byte_perm function
__global__ void exhaustiveSearchWithOneTableExtendedSharedMemoryBytePerm(u32* pt, u32* ct, u32* rk, u32* t0G, u32* t4G, u32* rconG, u64* range) {

	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int warpThreadIndex = threadIdx.x & 31;

	// <SHARED MEMORY>
	__shared__ u32 t0S[TABLE_SIZE][SHARED_MEM_BANK_SIZE];
	__shared__ u32 t4S[TABLE_SIZE];
	__shared__ u32 rconS[RCON_SIZE];
	__shared__ u32 ctS[U32_SIZE];

	if (threadIdx.x < TABLE_SIZE) {
		t4S[threadIdx.x] = t4G[threadIdx.x];
		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) {
			t0S[threadIdx.x][bankIndex] = t0G[threadIdx.x];
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

	u32 rk0Init, rk1Init, rk2Init, rk3Init;
	rk0Init = rk[0];
	rk1Init = rk[1];
	rk2Init = rk[2];
	rk3Init = rk[3];

	u32 pt0Init, pt1Init, pt2Init, pt3Init;
	pt0Init = pt[0];
	pt1Init = pt[1];
	pt2Init = pt[2];
	pt3Init = pt[3];

	u64 threadRange = *range;
	u64 threadRangeStart = (u64)threadIndex * threadRange;
	rk2Init = rk2Init + threadRangeStart / MAX_U32;
	rk3Init = rk3Init + threadRangeStart % MAX_U32;

	for (u64 rangeCount = 0; rangeCount < threadRange; rangeCount++) {

		u32 rk0, rk1, rk2, rk3;
		rk0 = rk0Init;
		rk1 = rk1Init;
		rk2 = rk2Init;
		rk3 = rk3Init;

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
		for (u8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1; roundCount++) {

			// Calculate round key
			u32 temp = rk3;
			rk0 = rk0 ^
				(t4S[(temp >> 16) & 0xff] & 0xff000000) ^
				(t4S[(temp >> 8) & 0xff] & 0x00ff0000) ^
				(t4S[(temp) & 0xff] & 0x0000ff00) ^
				(t4S[(temp >> 24)] & 0x000000ff) ^
				rconS[roundCount];
			rk1 = rk1 ^ rk0;
			rk2 = rk2 ^ rk1;
			rk3 = rk2 ^ rk3;

			// Table based round function
			t0 = t0S[s0 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s3 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk0;
			t1 = t0S[s1 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s0 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk1;
			t2 = t0S[s2 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s1 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk2;
			t3 = t0S[s3 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s2 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk3;

			s0 = t0;
			s1 = t1;
			s2 = t2;
			s3 = t3;

		}

		// Calculate the last round key
		u32 temp = rk3;
		rk0 = rk0 ^
			(t4S[(temp >> 16) & 0xff] & 0xff000000) ^
			(t4S[(temp >> 8) & 0xff] & 0x00ff0000) ^
			(t4S[(temp) & 0xff] & 0x0000ff00) ^
			(t4S[(temp >> 24)] & 0x000000ff) ^
			rconS[ROUND_COUNT_MIN_1];
		// Last round uses s-box directly and XORs to produce output.
		s0 = (t4S[t0 >> 24] & 0xFF000000) ^ (t4S[(t1 >> 16) & 0xff] & 0x00FF0000) ^ (t4S[(t2 >> 8) & 0xff] & 0x0000FF00) ^ (t4S[(t3) & 0xFF] & 0x000000FF) ^ rk0;
		if (s0 == ctS[0]) {
			rk1 = rk1 ^ rk0;
			s1 = (t4S[t1 >> 24] & 0xFF000000) ^ (t4S[(t2 >> 16) & 0xff] & 0x00FF0000) ^ (t4S[(t3 >> 8) & 0xff] & 0x0000FF00) ^ (t4S[(t0) & 0xFF] & 0x000000FF) ^ rk1;
			if (s1 == ctS[1]) {
				rk2 = rk2 ^ rk1;
				s2 = (t4S[t2 >> 24] & 0xFF000000) ^ (t4S[(t3 >> 16) & 0xff] & 0x00FF0000) ^ (t4S[(t0 >> 8) & 0xff] & 0x0000FF00) ^ (t4S[(t1) & 0xFF] & 0x000000FF) ^ rk2;
				if (s2 == ctS[2]) {
					rk3 = rk2 ^ rk3;
					s3 = (t4S[t3 >> 24] & 0xFF000000) ^ (t4S[(t0 >> 16) & 0xff] & 0x00FF0000) ^ (t4S[(t1 >> 8) & 0xff] & 0x0000FF00) ^ (t4S[(t2) & 0xFF] & 0x000000FF) ^ rk3;
					if (s3 == ctS[3]) {
						printf("! Found key : %08x %08x %08x %08x\n", rk0Init, rk1Init, rk2Init, rk3Init);
						printf("-------------------------------\n");
					}
				}
			}
		}

		// Overflow
		if (rk3Init == MAX_U32) {
			rk2Init++;
		}

		// Create key as 32 bit unsigned integers
		rk3Init++;
	}
}

// Exhaustive search with one table extended as 32 columns
// 1 Table [256][32] -> arithmetic shift: __byte_perm function
// SBox[256] is partly expanded
__global__ void exhaustiveSearchWithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBox(u32* pt, u32* ct, u32* rk, u32* t0G, u32* t4G, u32* rconG, u64* range) {

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

	u32 rk0Init, rk1Init, rk2Init, rk3Init;
	rk0Init = rk[0];
	rk1Init = rk[1];
	rk2Init = rk[2];
	rk3Init = rk[3];

	u32 pt0Init, pt1Init, pt2Init, pt3Init;
	pt0Init = pt[0];
	pt1Init = pt[1];
	pt2Init = pt[2];
	pt3Init = pt[3];

	u64 threadRange = *range;
	u64 threadRangeStart = (u64)threadIndex * threadRange;
	rk2Init = rk2Init + threadRangeStart / MAX_U32;
	rk3Init = rk3Init + threadRangeStart % MAX_U32;

	for (u64 rangeCount = 0; rangeCount < threadRange; rangeCount++) {

		u32 rk0, rk1, rk2, rk3;
		rk0 = rk0Init;
		rk1 = rk1Init;
		rk2 = rk2Init;
		rk3 = rk3Init;

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
		for (u8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1; roundCount++) {

			// Calculate round key
			u32 temp = rk3;
			rk0 = rk0 ^
				(t4S[(temp >> 16) & 0xff][warpThreadIndexSBox] & 0xff000000) ^
				(t4S[(temp >> 8) & 0xff][warpThreadIndexSBox] & 0x00ff0000) ^
				(t4S[(temp) & 0xff][warpThreadIndexSBox] & 0x0000ff00) ^
				(t4S[(temp >> 24)][warpThreadIndexSBox] & 0x000000ff) ^
				rconS[roundCount];
			rk1 = rk1 ^ rk0;
			rk2 = rk2 ^ rk1;
			rk3 = rk2 ^ rk3;

			// Table based round function
			t0 = t0S[s0 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s3 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk0;
			t1 = t0S[s1 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s0 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk1;
			t2 = t0S[s2 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s1 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk2;
			t3 = t0S[s3 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s2 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk3;

			s0 = t0;
			s1 = t1;
			s2 = t2;
			s3 = t3;

		}

		// Calculate the last round key
		u32 temp = rk3;
		rk0 = rk0 ^
			(t4S[(temp >> 16) & 0xff][warpThreadIndexSBox] & 0xff000000) ^
			(t4S[(temp >> 8) & 0xff][warpThreadIndexSBox] & 0x00ff0000) ^
			(t4S[(temp) & 0xff][warpThreadIndexSBox] & 0x0000ff00) ^
			(t4S[(temp >> 24)][warpThreadIndexSBox] & 0x000000ff) ^
			rconS[ROUND_COUNT_MIN_1];
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
						printf("! Found key : %08x %08x %08x %08x\n", rk0Init, rk1Init, rk2Init, rk3Init);
						printf("-------------------------------\n");
					}
				}
			}
		}

		// Overflow
		if (rk3Init == MAX_U32) {
			rk2Init++;
		}

		// Create key as 32 bit unsigned integers
		rk3Init++;
	}
}
__global__ void exhaustiveSearchWithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBoxCihangir(u32* pt, u32* ct, u32* rk, u32* t0G, u32* t4G, u32* rconG, u64* range, u8 *SAES) {
	u64 threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int warpThreadIndex = threadIdx.x & 31;
//	int warpThreadIndexSBox = warpThreadIndex % S_BOX_BANK_SIZE;
	// <SHARED MEMORY>
	__shared__ u32 t0S[TABLE_SIZE][SHARED_MEM_BANK_SIZE];
	__shared__ u32 rconS[RCON_SIZE];
	__shared__ u32 ctS[U32_SIZE];
//	__shared__ u32 t4S[TABLE_SIZE][S_BOX_BANK_SIZE];
	__shared__ u8 Sbox[64][32][4];
	
	if (threadIdx.x < TABLE_SIZE) {
		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) {			
			t0S[threadIdx.x][bankIndex] = t0G[threadIdx.x];
			Sbox[threadIdx.x / 4][bankIndex][threadIdx.x % 4] = SAES[threadIdx.x];
		}
//		for (u8 bankIndex = 0; bankIndex < S_BOX_BANK_SIZE; bankIndex++) {			t4S[threadIdx.x][bankIndex] = t4G[threadIdx.x];		}
//		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) { Sbox[threadIdx.x/4][bankIndex][threadIdx.x % 4] = SAES[threadIdx.x]; }
		if (threadIdx.x < U32_SIZE) { ctS[threadIdx.x] = ct[threadIdx.x]; }
		if (threadIdx.x < RCON_SIZE) {			rconS[threadIdx.x] = rconG[threadIdx.x];		}
		
	}	// </SHARED MEMORY>
	__syncthreads(); // Wait until every thread is ready
	u32 rk0Init, rk1Init, rk2Init, rk3Init;
	rk0Init = rk[0];	rk1Init = rk[1];	rk2Init = rk[2];	rk3Init = rk[3];
	u32 pt0Init, pt1Init, pt2Init, pt3Init;
	pt0Init = pt[0];	pt1Init = pt[1];	pt2Init = pt[2];	pt3Init = pt[3];
	u64 threadRange = *range;
	u64 threadRangeStart = threadIndex * threadRange;
	rk2Init = rk2Init + threadRangeStart / (u64)MAX_U32;
	rk3Init = rk3Init + threadRangeStart % (u64)MAX_U32;
	for (u64 rangeCount = 0; rangeCount < threadRange; rangeCount++) {
		u32 rk0, rk1, rk2, rk3;
		rk0 = rk0Init;		rk1 = rk1Init;		rk2 = rk2Init;		rk3 = rk3Init;
		// Create plaintext as 32 bit unsigned integers
		u32 s0, s1, s2, s3;
		s0 = pt0Init;		s1 = pt1Init;		s2 = pt2Init;		s3 = pt3Init;
		// First round just XORs input with key.
		s0 = s0 ^ rk0;		s1 = s1 ^ rk1;		s2 = s2 ^ rk2;		s3 = s3 ^ rk3;
		u32 t0, t1, t2, t3;
		for (u8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1; roundCount++) {
			// Calculate round key
			u32 temp = rk3;
			rk0 = rk0 ^
				arithmeticRightShiftBytePerm((u32)Sbox[((temp >> 16) & 0xff) / 4][warpThreadIndex][((temp >> 16)) % 4], SHIFT_1_RIGHT) ^
				arithmeticRightShiftBytePerm((u32)Sbox[((temp >> 8) & 0xff) / 4][warpThreadIndex][((temp >> 8)) % 4], SHIFT_2_RIGHT) ^
				arithmeticRightShiftBytePerm((u32)Sbox[((temp) & 0xff) / 4][warpThreadIndex][((temp)) % 4], SHIFT_3_RIGHT) ^
				((u32)Sbox[((temp >> 24) / 4)][warpThreadIndex][((temp >> 24) % 4)]) ^
				rconS[roundCount];
			rk1 = rk1 ^ rk0;			rk2 = rk2 ^ rk1;			rk3 = rk2 ^ rk3;
			// Table based round function
			t0 = t0S[s0 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s3 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk0;
			t1 = t0S[s1 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s0 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk1;
			t2 = t0S[s2 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s1 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk2;
			t3 = t0S[s3 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s2 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk3;
			s0 = t0;			s1 = t1;			s2 = t2;			s3 = t3;
		}
		// Calculate the last round key
		u32 temp = rk3;
		rk0 = rk0 ^
			arithmeticRightShiftBytePerm((u32)Sbox[((temp >> 16) & 0xff) / 4][warpThreadIndex][((temp >> 16)) % 4], SHIFT_1_RIGHT) ^
			arithmeticRightShiftBytePerm((u32)Sbox[((temp >> 8) & 0xff) / 4][warpThreadIndex][((temp >> 8)) % 4], SHIFT_2_RIGHT) ^
			arithmeticRightShiftBytePerm((u32)Sbox[((temp) & 0xff) / 4][warpThreadIndex][((temp)) % 4], SHIFT_3_RIGHT) ^
			((u32)Sbox[((temp >> 24) / 4)][warpThreadIndex][((temp >> 24) % 4)]) ^
			rconS[ROUND_COUNT_MIN_1];
		// Last round uses s-box directly and XORs to produce output.
		s0 = arithmeticRightShiftBytePerm((u32)Sbox[((t0 >> 24)) / 4][warpThreadIndex][((t0 >> 24)) % 4], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t1 >> 16) & 0xff) / 4][warpThreadIndex][((t1 >> 16)) % 4], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t2 >> 8) &0xFF) / 4][warpThreadIndex][((t2 >> 8)) % 4], SHIFT_3_RIGHT) ^ ((u32)Sbox[((t3 & 0xFF) / 4)][warpThreadIndex][((t3 & 0xFF) % 4)]) ^ rk0;
		if (s0 == ctS[0]) {
			rk1 = rk1 ^ rk0;
			s1 = arithmeticRightShiftBytePerm((u32)Sbox[((t1 >> 24)) / 4][warpThreadIndex][((t1 >> 24)) % 4], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t2 >> 16) & 0xff) / 4][warpThreadIndex][((t2 >> 16)) % 4], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t3 >> 8) & 0xFF) / 4][warpThreadIndex][((t3 >> 8)) % 4], SHIFT_3_RIGHT) ^ ((u32)Sbox[((t0 & 0xFF) / 4)][warpThreadIndex][((t0 & 0xFF) % 4)]) ^ rk1;
			if (s1 == ctS[1]) {
				rk2 = rk2 ^ rk1;
				s2 = arithmeticRightShiftBytePerm((u32)Sbox[((t2 >> 24)) / 4][warpThreadIndex][((t2 >> 24)) % 4], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t3 >> 16) & 0xff) / 4][warpThreadIndex][((t3 >> 16)) % 4], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t0 >> 8) & 0xFF) / 4][warpThreadIndex][((t0 >> 8)) % 4], SHIFT_3_RIGHT) ^ ((u32)Sbox[((t1 & 0xFF) / 4)][warpThreadIndex][((t1 & 0xFF) % 4)]) ^ rk2;
				if (s2 == ctS[2]) {
					rk3 = rk2 ^ rk3;
					s3 = arithmeticRightShiftBytePerm((u32)Sbox[((t3 >> 24)) / 4][warpThreadIndex][((t3 >> 24)) % 4], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t0 >> 16) & 0xff) / 4][warpThreadIndex][((t0 >> 16)) % 4], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t1 >> 8) & 0xFF) / 4][warpThreadIndex][((t1 >> 8)) % 4], SHIFT_3_RIGHT) ^ ((u32)Sbox[((t2 & 0xFF) / 4)][warpThreadIndex][((t2 & 0xFF) % 4)]) ^ rk3;
					if (s3 == ctS[3]) {
						printf("! Found key : %08x %08x %08x %08x\n", rk0Init, rk1Init, rk2Init, rk3Init);
						printf("-------------------------------\n");
					}
				}
			}
		}
		// Overflow
		if (rk3Init == MAX_U32) {			rk2Init++;		}
		rk3Init++;		// Create key as 32 bit unsigned integers
	}
}
__global__ void exhaustiveSearchWithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBoxCihangir2(u32* pt, u32* ct, u32* rk, u32* t0G, u32* t4G, u32* rconG, u64* range, u8* SAES) {
	u64 threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int warpThreadIndex = threadIdx.x & 31;
	//	int warpThreadIndexSBox = warpThreadIndex % S_BOX_BANK_SIZE;
		// <SHARED MEMORY>
	__shared__ u32 t0S[TABLE_SIZE][SHARED_MEM_BANK_SIZE];
	__shared__ u32 rconS[RCON_SIZE];
	__shared__ u32 ctS[U32_SIZE];
	//	__shared__ u32 t4S[TABLE_SIZE][S_BOX_BANK_SIZE];
	__shared__ u8 Sbox[64][32][4];

	if (threadIdx.x < TABLE_SIZE) {
		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) {
			t0S[threadIdx.x][bankIndex] = t0G[threadIdx.x];
			Sbox[threadIdx.x / 4][bankIndex][threadIdx.x % 4] = SAES[threadIdx.x];
		}
		//		for (u8 bankIndex = 0; bankIndex < S_BOX_BANK_SIZE; bankIndex++) {			t4S[threadIdx.x][bankIndex] = t4G[threadIdx.x];		}
		//		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) { Sbox[threadIdx.x/4][bankIndex][threadIdx.x % 4] = SAES[threadIdx.x]; }
		if (threadIdx.x < U32_SIZE) { ctS[threadIdx.x] = ct[threadIdx.x]; }
		if (threadIdx.x < RCON_SIZE) { rconS[threadIdx.x] = rconG[threadIdx.x]; }

	}	// </SHARED MEMORY>
	__syncthreads(); // Wait until every thread is ready
	u32 rk0Init, rk1Init, rk2Init, rk3Init;
	rk0Init = rk[0];	rk1Init = rk[1];	rk2Init = rk[2];	rk3Init = rk[3];
	u32 pt0Init, pt1Init, pt2Init, pt3Init;
	pt0Init = pt[0];	pt1Init = pt[1];	pt2Init = pt[2];	pt3Init = pt[3];
	u64 threadRange = *range;
	u64 threadRangeStart = threadIndex * threadRange;
	rk2Init = rk2Init + threadRangeStart / (u64)MAX_U32;
	rk3Init = rk3Init + threadRangeStart % (u64)MAX_U32;
	for (u64 rangeCount = 0; rangeCount < threadRange; rangeCount++) {
		u32 rk0, rk1, rk2, rk3;
		rk0 = rk0Init;		rk1 = rk1Init;		rk2 = rk2Init;		rk3 = rk3Init;
		// Create plaintext as 32 bit unsigned integers
		u32 s0, s1, s2, s3;
		s0 = pt0Init;		s1 = pt1Init;		s2 = pt2Init;		s3 = pt3Init;
		// First round just XORs input with key.
		s0 = s0 ^ rk0;		s1 = s1 ^ rk1;		s2 = s2 ^ rk2;		s3 = s3 ^ rk3;
		u32 t0, t1, t2, t3;
		for (u8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1; roundCount++) {
			// Calculate round key
			u32 temp = rk3;
			rk0 = rk0 ^
				arithmeticRightShift((u32)Sbox[((temp >> 16) & 0xff) / 4][warpThreadIndex][((temp >> 16)) % 4], 8) ^
				arithmeticRightShift((u32)Sbox[((temp >> 8) & 0xff) / 4][warpThreadIndex][((temp >> 8)) % 4], 16) ^
				arithmeticRightShift((u32)Sbox[((temp) & 0xff) / 4][warpThreadIndex][((temp)) % 4], 24) ^
				((u32)Sbox[((temp >> 24) / 4)][warpThreadIndex][((temp >> 24) % 4)]) ^
				rconS[roundCount];
			rk1 = rk1 ^ rk0;			rk2 = rk2 ^ rk1;			rk3 = rk2 ^ rk3;
			// Table based round function
			t0 = t0S[s0 >> 24][warpThreadIndex] ^ arithmeticRightShift(t0S[(s1 >> 16) & 0xFF][warpThreadIndex], 8) ^ arithmeticRightShift(t0S[(s2 >> 8) & 0xFF][warpThreadIndex], 16) ^ arithmeticRightShift(t0S[s3 & 0xFF][warpThreadIndex], 24) ^ rk0;
			t1 = t0S[s1 >> 24][warpThreadIndex] ^ arithmeticRightShift(t0S[(s2 >> 16) & 0xFF][warpThreadIndex], 8) ^ arithmeticRightShift(t0S[(s3 >> 8) & 0xFF][warpThreadIndex], 16) ^ arithmeticRightShift(t0S[s0 & 0xFF][warpThreadIndex], 24) ^ rk1;
			t2 = t0S[s2 >> 24][warpThreadIndex] ^ arithmeticRightShift(t0S[(s3 >> 16) & 0xFF][warpThreadIndex], 8) ^ arithmeticRightShift(t0S[(s0 >> 8) & 0xFF][warpThreadIndex], 16) ^ arithmeticRightShift(t0S[s1 & 0xFF][warpThreadIndex], 24) ^ rk2;
			t3 = t0S[s3 >> 24][warpThreadIndex] ^ arithmeticRightShift(t0S[(s0 >> 16) & 0xFF][warpThreadIndex], 8) ^ arithmeticRightShift(t0S[(s1 >> 8) & 0xFF][warpThreadIndex], 16) ^ arithmeticRightShift(t0S[s2 & 0xFF][warpThreadIndex], 24) ^ rk3;
			s0 = t0;			s1 = t1;			s2 = t2;			s3 = t3;
		}
		// Calculate the last round key
		u32 temp = rk3;
		rk0 = rk0 ^
			arithmeticRightShift((u32)Sbox[((temp >> 16) & 0xff) / 4][warpThreadIndex][((temp >> 16)) % 4], 8) ^
			arithmeticRightShift((u32)Sbox[((temp >> 8) & 0xff) / 4][warpThreadIndex][((temp >> 8)) % 4], 16) ^
			arithmeticRightShift((u32)Sbox[((temp) & 0xff) / 4][warpThreadIndex][((temp)) % 4], 24) ^
			((u32)Sbox[((temp >> 24) / 4)][warpThreadIndex][((temp >> 24) % 4)]) ^
			rconS[ROUND_COUNT_MIN_1];
		// Last round uses s-box directly and XORs to produce output.
		s0 = arithmeticRightShift((u32)Sbox[((t0 >> 24)) / 4][warpThreadIndex][((t0 >> 24)) % 4], 8) ^ arithmeticRightShift((u32)Sbox[((t1 >> 16) & 0xff) / 4][warpThreadIndex][((t1 >> 16)) % 4], 16) ^ arithmeticRightShift((u32)Sbox[((t2 >> 8) & 0xFF) / 4][warpThreadIndex][((t2 >> 8)) % 4], 24) ^ ((u32)Sbox[((t3 & 0xFF) / 4)][warpThreadIndex][((t3 & 0xFF) % 4)]) ^ rk0;
		if (s0 == ctS[0]) {
			rk1 = rk1 ^ rk0;
			s1 = arithmeticRightShift((u32)Sbox[((t1 >> 24)) / 4][warpThreadIndex][((t1 >> 24)) % 4], 8) ^ arithmeticRightShift((u32)Sbox[((t2 >> 16) & 0xff) / 4][warpThreadIndex][((t2 >> 16)) % 4], 16) ^ arithmeticRightShift((u32)Sbox[((t3 >> 8) & 0xFF) / 4][warpThreadIndex][((t3 >> 8)) % 4], 24) ^ ((u32)Sbox[((t0 & 0xFF) / 4)][warpThreadIndex][((t0 & 0xFF) % 4)]) ^ rk1;
			if (s1 == ctS[1]) {
				rk2 = rk2 ^ rk1;
				s2 = arithmeticRightShift((u32)Sbox[((t2 >> 24)) / 4][warpThreadIndex][((t2 >> 24)) % 4], 8) ^ arithmeticRightShift((u32)Sbox[((t3 >> 16) & 0xff) / 4][warpThreadIndex][((t3 >> 16)) % 4], 16) ^ arithmeticRightShift((u32)Sbox[((t0 >> 8) & 0xFF) / 4][warpThreadIndex][((t0 >> 8)) % 4], 24) ^ ((u32)Sbox[((t1 & 0xFF) / 4)][warpThreadIndex][((t1 & 0xFF) % 4)]) ^ rk2;
				if (s2 == ctS[2]) {
					rk3 = rk2 ^ rk3;
					s3 = arithmeticRightShift((u32)Sbox[((t3 >> 24)) / 4][warpThreadIndex][((t3 >> 24)) % 4], 8) ^ arithmeticRightShift((u32)Sbox[((t0 >> 16) & 0xff) / 4][warpThreadIndex][((t0 >> 16)) % 4], 16) ^ arithmeticRightShift((u32)Sbox[((t1 >> 8) & 0xFF) / 4][warpThreadIndex][((t1 >> 8)) % 4], 24) ^ ((u32)Sbox[((t2 & 0xFF) / 4)][warpThreadIndex][((t2 & 0xFF) % 4)]) ^ rk3;
					if (s3 == ctS[3]) {
						printf("! Found key : %08x %08x %08x %08x\n", rk0Init, rk1Init, rk2Init, rk3Init);
						printf("-------------------------------\n");
					}
				}
			}
		}
		// Overflow
		if (rk3Init == MAX_U32) { rk2Init++; }
		rk3Init++;		// Create key as 32 bit unsigned integers
	}
}
/*__global__ void exhaustiveSearchCem(u32* pt, u32* ct, u32* rk, u32* t0G, u32* t1G, u32* t4G, u32* rconG, u64* range, u8* SAES) {
	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int warpThreadIndex = threadIdx.x & 31;
	//	int warpThreadIndexSBox = warpThreadIndex % S_BOX_BANK_SIZE;
		// <SHARED MEMORY>
	__shared__ u32 t0S[TABLE_SIZE][SHARED_MEM_BANK_SIZE];
	__shared__ u32 t1S[TABLE_SIZE][SHARED_MEM_BANK_SIZE];
	//	__shared__ u32 t4S[TABLE_SIZE][S_BOX_BANK_SIZE];
	__shared__ u8 Sbox[64][32][4];
	__shared__ u32 rconS[RCON_SIZE];
	__shared__ u32 ctS[U32_SIZE];
	if (threadIdx.x < TABLE_SIZE) {
		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) { t0S[threadIdx.x][bankIndex] = t0G[threadIdx.x]; }
		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) { t1S[threadIdx.x][bankIndex] = t1G[threadIdx.x]; }
		//		for (u8 bankIndex = 0; bankIndex < S_BOX_BANK_SIZE; bankIndex++) {			t4S[threadIdx.x][bankIndex] = t4G[threadIdx.x];		}
		for (u8 bankIndex = 0; bankIndex < S_BOX_BANK_SIZE; bankIndex++) { Sbox[threadIdx.x / 4][bankIndex][threadIdx.x % 4] = SAES[threadIdx.x]; }
		if (threadIdx.x < RCON_SIZE) { rconS[threadIdx.x] = rconG[threadIdx.x]; }
		if (threadIdx.x < U32_SIZE) { ctS[threadIdx.x] = ct[threadIdx.x]; }
	}	// </SHARED MEMORY>
	__syncthreads(); // Wait until every thread is ready
	u32 rk0Init, rk1Init, rk2Init, rk3Init;
	rk0Init = rk[0];	rk1Init = rk[1];	rk2Init = rk[2];	rk3Init = rk[3];
	u32 pt0Init, pt1Init, pt2Init, pt3Init;
	pt0Init = pt[0];	pt1Init = pt[1];	pt2Init = pt[2];	pt3Init = pt[3];
	u64 threadRange = *range;
	u64 threadRangeStart = (u64)threadIndex * threadRange;
	rk2Init = rk2Init + threadRangeStart / MAX_U32;
	rk3Init = rk3Init + threadRangeStart % MAX_U32;
	for (u64 rangeCount = 0; rangeCount < threadRange; rangeCount++) {
		u32 rk0, rk1, rk2, rk3;
		rk0 = rk0Init;		rk1 = rk1Init;		rk2 = rk2Init;		rk3 = rk3Init;
		// Create plaintext as 32 bit unsigned integers
		u32 s0, s1, s2, s3;
		s0 = pt0Init;		s1 = pt1Init;		s2 = pt2Init;		s3 = pt3Init;
		// First round just XORs input with key.
		s0 = s0 ^ rk0;		s1 = s1 ^ rk1;		s2 = s2 ^ rk2;		s3 = s3 ^ rk3;
		u32 t0, t1, t2, t3;
		for (u8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1; roundCount++) {
			// Calculate round key
			u32 temp = rk3;
			rk0 = rk0 ^
				arithmeticRightShiftBytePerm((u64)Sbox[((temp >> 16) & 0xff) / 4][warpThreadIndex][((temp >> 16) & 0xff) % 4], SHIFT_1_RIGHT) ^
				arithmeticRightShiftBytePerm((u64)Sbox[((temp >> 8) & 0xff) / 4][warpThreadIndex][((temp >> 8) & 0xff) % 4], SHIFT_2_RIGHT) ^
				arithmeticRightShiftBytePerm((u64)Sbox[((temp) & 0xff) / 4][warpThreadIndex][((temp) & 0xff) % 4], SHIFT_3_RIGHT) ^
				((u64)Sbox[((temp >> 24) / 4)][warpThreadIndex][((temp >> 24) % 4)]) ^
				rconS[roundCount];

			rk1 = rk1 ^ rk0;			rk2 = rk2 ^ rk1;			rk3 = rk2 ^ rk3;
			// Table based round function
			t0 = t0S[s0 >> 24][warpThreadIndex] ^ t1S[(s1 >> 16) & 0xFF][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s3 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk0;
			t1 = t0S[s1 >> 24][warpThreadIndex] ^ t1S[(s2 >> 16) & 0xFF][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s0 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk1;
			t2 = t0S[s2 >> 24][warpThreadIndex] ^ t1S[(s3 >> 16) & 0xFF][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s1 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk2;
			t3 = t0S[s3 >> 24][warpThreadIndex] ^ t1S[(s0 >> 16) & 0xFF][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s2 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk3;
			s0 = t0;			s1 = t1;			s2 = t2;			s3 = t3;
		}
		// Calculate the last round key
		u32 temp = rk3;
		rk0 = rk0 ^
			arithmeticRightShiftBytePerm((u64)Sbox[((temp >> 16) & 0xff) / 4][warpThreadIndex][((temp >> 16)) % 4], SHIFT_1_RIGHT) ^
			arithmeticRightShiftBytePerm((u64)Sbox[((temp >> 8) & 0xff) / 4][warpThreadIndex][((temp >> 8)) % 4], SHIFT_2_RIGHT) ^
			arithmeticRightShiftBytePerm((u64)Sbox[((temp) & 0xff) / 4][warpThreadIndex][((temp)) % 4], SHIFT_3_RIGHT) ^
			((u64)Sbox[((temp >> 24) / 4)][warpThreadIndex][((temp >> 24) % 4)]) ^
			rconS[ROUND_COUNT_MIN_1];
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
						printf("! Found key : %08x %08x %08x %08x\n", rk0Init, rk1Init, rk2Init, rk3Init);
						printf("-------------------------------\n");
					}
				}
			}
		}
		// Overflow
		if (rk3Init == MAX_U32) { rk2Init++; }
		rk3Init++;		// Create key as 32 bit unsigned integers
	}
}*/


// Exhaustive search with one table extended as 32 columns
// 1 Table [256][32] -> arithmetic shift: __byte_perm function
// 4 S-box, each shifted
__global__ void exhaustiveSearchWithOneTableExtendedSharedMemoryBytePerm4ShiftedSbox(u32* pt, u32* ct, u32* rk, u32* t0G, u32* t4_0G, u32* t4_1G, u32* t4_2G, u32* t4_3G, u32* rconG, u64* range) {

	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int warpThreadIndex = threadIdx.x & 31;

	// <SHARED MEMORY>
	__shared__ u32 t0S[TABLE_SIZE][SHARED_MEM_BANK_SIZE];
	__shared__ u32 t4_0S[TABLE_SIZE];
	__shared__ u32 t4_1S[TABLE_SIZE];
	__shared__ u32 t4_2S[TABLE_SIZE];
	__shared__ u32 t4_3S[TABLE_SIZE];
	__shared__ u32 rconS[RCON_SIZE];
	__shared__ u32 ctS[U32_SIZE];

	if (threadIdx.x < TABLE_SIZE) {
		t4_0S[threadIdx.x] = t4_0G[threadIdx.x];
		t4_1S[threadIdx.x] = t4_1G[threadIdx.x];
		t4_2S[threadIdx.x] = t4_2G[threadIdx.x];
		t4_3S[threadIdx.x] = t4_3G[threadIdx.x];
		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) {
			t0S[threadIdx.x][bankIndex] = t0G[threadIdx.x];
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

	u32 rk0Init, rk1Init, rk2Init, rk3Init;
	rk0Init = rk[0];
	rk1Init = rk[1];
	rk2Init = rk[2];
	rk3Init = rk[3];

	u32 pt0Init, pt1Init, pt2Init, pt3Init;
	pt0Init = pt[0];
	pt1Init = pt[1];
	pt2Init = pt[2];
	pt3Init = pt[3];

	u64 threadRange = *range;
	u64 threadRangeStart = (u64)threadIndex * threadRange;
	rk2Init = rk2Init + threadRangeStart / MAX_U32;
	rk3Init = rk3Init + threadRangeStart % MAX_U32;

	for (u64 rangeCount = 0; rangeCount < threadRange; rangeCount++) {

		u32 rk0, rk1, rk2, rk3;
		rk0 = rk0Init;
		rk1 = rk1Init;
		rk2 = rk2Init;
		rk3 = rk3Init;

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
		for (u8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1; roundCount++) {

			// Calculate round key
			u32 temp = rk3;
			rk0 = rk0 ^ t4_3S[(temp >> 16) & 0xff] ^ t4_2S[(temp >> 8) & 0xff] ^ t4_1S[(temp) & 0xff] ^ t4_0S[(temp >> 24)] ^ rconS[roundCount];
			rk1 = rk1 ^ rk0;
			rk2 = rk2 ^ rk1;
			rk3 = rk2 ^ rk3;

			// Table based round function
			t0 = t0S[s0 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s3 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk0;
			t1 = t0S[s1 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s0 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk1;
			t2 = t0S[s2 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s1 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk2;
			t3 = t0S[s3 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s2 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk3;

			s0 = t0;
			s1 = t1;
			s2 = t2;
			s3 = t3;

		}

		// Calculate the last round key
		u32 temp = rk3;
		rk0 = rk0 ^ t4_3S[(temp >> 16) & 0xff] ^ t4_2S[(temp >> 8) & 0xff] ^ t4_1S[(temp) & 0xff] ^ t4_0S[(temp >> 24)] ^ rconS[ROUND_COUNT_MIN_1];
		// Last round uses s-box directly and XORs to produce output.
		s0 = t4_3S[t0 >> 24] ^ t4_2S[(t1 >> 16) & 0xff] ^ t4_1S[(t2 >> 8) & 0xff] ^ t4_0S[(t3) & 0xFF] ^ rk0;
		if (s0 == ctS[0]) {
			rk1 = rk1 ^ rk0;
			s1 = t4_3S[t1 >> 24] ^ t4_2S[(t2 >> 16) & 0xff] ^ t4_1S[(t3 >> 8) & 0xff] ^ t4_0S[(t0) & 0xFF] ^ rk1;
			if (s1 == ctS[1]) {
				rk2 = rk2 ^ rk1;
				s2 = t4_3S[t2 >> 24] ^ t4_2S[(t3 >> 16) & 0xff] ^ t4_1S[(t0 >> 8) & 0xff] ^ t4_0S[(t1) & 0xFF] ^ rk2;
				if (s2 == ctS[2]) {
					rk3 = rk2 ^ rk3;
					s3 = t4_3S[t3 >> 24] ^ t4_2S[(t0 >> 16) & 0xff] ^ t4_1S[(t1 >> 8) & 0xff] ^ t4_0S[(t2) & 0xFF] ^ rk3;
					if (s3 == ctS[3]) {
						printf("! Found key : %08x %08x %08x %08x\n", rk0Init, rk1Init, rk2Init, rk3Init);
						printf("-------------------------------\n");
					}
				}
			}
		}

		// Overflow
		if (rk3Init == MAX_U32) {
			rk2Init++;
		}

		// Create key as 32 bit unsigned integers
		rk3Init++;
	}
}

__host__ int main128ExhaustiveSearch(int choice) {
	printf("\n");	printf("########## AES-128 Exhaustive Search Implementation ##########\n");	printf("\n");
	// Allocate plaintext, ciphertext and initial round key
	u32 *pt, *ct, *rk;
	gpuErrorCheck(cudaMallocManaged(&pt, 4 * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&ct, 4 * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&rk, 4 * sizeof(u32)));
	pt[0] = 0x3243F6A8U;	pt[1] = 0x885A308DU;	pt[2] = 0x313198A2U;	pt[3] = 0xE0370734U;
//	pt[0] = 0;	pt[1] = 0;	pt[2] = 0;	pt[3] = 0;
	ct[0] = 0x3925841DU;	ct[1] = 0x02DC09FBU;	ct[2] = 0xDC118597U;	ct[3] = 0x196A0B32U;
	// aes-cipher-internals.xlsx
	rk[0] = 0x2B7E1516U;	rk[1] = 0x28AED2A6U;	rk[2] = 0xABF71588U;	rk[3] = 0x09CF0000U;
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
	for (int i = 0; i < TABLE_SIZE; i++) {
		t0[i] = T0[i];		t1[i] = T1[i];		t2[i] = T2[i];		t3[i] = T3[i];		t4[i] = T4[i];
		t4_0[i] = T4_0[i];		t4_1[i] = T4_1[i];		t4_2[i] = T4_2[i];		t4_3[i] = T4_3[i];
	}
	for (int i = 0; i < 256; i++) SAES_d[i] = SAES[i]; // Cihangir
	printf("-------------------------------\n");
	u64* range = calculateRange();
/*	printf("Plaintext                     : %08x %08x %08x %08x\n", pt[0], pt[1], pt[2], pt[3]);
	printf("Ciphertext                    : %08x %08x %08x %08x\n", ct[0], ct[1], ct[2], ct[3]);
	printf("Initial Key                   : %08x %08x %08x %08x\n", rk[0], rk[1], rk[2], rk[3]);
	printf("-------------------------------\n");*/



	clock_t beginTime = clock();
//	exhaustiveSearch << <BLOCKS, THREADS >> > (pt, ct, rk, t0, t1, t2, t3, t4, rcon, range);
//	exhaustiveSearchWithOneTable<<<BLOCKS, THREADS>>>(pt, ct, rk, t0, t4, rcon, range);
	if (choice == 1) exhaustiveSearchWithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBoxCihangir2 << <BLOCKS, THREADS >> > (pt, ct, rk, t0, t4, rcon, range, SAES_d);
	if (choice == 11) exhaustiveSearchWithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBoxCihangir << <BLOCKS, THREADS >> > (pt, ct, rk, t0, t4, rcon, range, SAES_d);
//	else if (choice == 2) exhaustiveSearchWithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBox << <BLOCKS, THREADS >> > (pt, ct, rk, t0, t4, rcon, range);
	else if (choice == 2) exhaustiveSearchWithOneTableExtendedSharedMemory << <BLOCKS, THREADS >> > (pt, ct, rk, t0, t4, rcon, range);
	else if (choice == 22) exhaustiveSearchWithOneTableExtendedSharedMemoryBytePerm << <BLOCKS, THREADS >> > (pt, ct, rk, t0, t4, rcon, range);
	
//	exhaustiveSearchWithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBox << <BLOCKS, THREADS >> > (pt, ct, rk, t0, t4, rcon, range);
//	exhaustiveSearchCem << <BLOCKS, THREADS >> > (pt, ct, rk, t0, t1, t4, rcon, range, SAES_d);
	cudaDeviceSynchronize();
	printf("Time elapsed: %f sec\n", float(clock() - beginTime) / CLOCKS_PER_SEC);

/*	beginTime = clock();
	// Kernels
	//exhaustiveSearch<<<BLOCKS, THREADS>>>(pt, ct, rk, t0, t1, t2, t3, t4, rcon, range);
	//exhaustiveSearchWithOneTable<<<BLOCKS, THREADS>>>(pt, ct, rk, t0, t4, rcon, range);
	//exhaustiveSearchWithOneTableExtendedSharedMemory<<<BLOCKS, THREADS>>>(pt, ct, rk, t0, t4, rcon, range);
	//exhaustiveSearchWithOneTableExtendedSharedMemoryBytePerm<<<BLOCKS, THREADS>>>(pt, ct, rk, t0, t4, rcon, range);
	// Fastest
	exhaustiveSearchWithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBox<<<BLOCKS, THREADS>>>(pt, ct, rk, t0, t4, rcon, range);
	//exhaustiveSearchWithOneTableExtendedSharedMemoryBytePerm4ShiftedSbox<<<BLOCKS, THREADS>>>(pt, ct, rk, t0, t4_0, t4_1, t4_2, t4_3, rcon, range);
	cudaDeviceSynchronize();
	printf("Time elapsed: %f sec\n", float(clock() - beginTime) / CLOCKS_PER_SEC);*/


	printf("-------------------------------\n");
	printLastCUDAError();
	// Free alocated arrays
	cudaFree(range); cudaFree(pt);	cudaFree(ct);	cudaFree(rk);	cudaFree(t0);	cudaFree(t1);	cudaFree(t2);	cudaFree(t3);	cudaFree(t4);	
	cudaFree(t4_0);	cudaFree(t4_1);	cudaFree(t4_2);	cudaFree(t4_3);	cudaFree(rcon); cudaFree(SAES_d);
	return 0;
}

