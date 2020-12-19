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


// Key expansion from given key set, populate rk[44]
__host__ void keyExpansion(u32* key, u32* rk) {

	u32 rk0, rk1, rk2, rk3;
	rk0 = key[0];
	rk1 = key[1];
	rk2 = key[2];
	rk3 = key[3];

	rk[0] = rk0;
	rk[1] = rk1;
	rk[2] = rk2;
	rk[3] = rk3;

	for (u8 roundCount = 0; roundCount < ROUND_COUNT; roundCount++) {
		u32 temp = rk3;
		rk0 = rk0 ^ T4_3[(temp >> 16) & 0xff] ^ T4_2[(temp >> 8) & 0xff] ^ T4_1[(temp) & 0xff] ^ T4_0[(temp >> 24)] ^ RCON32[roundCount];
		rk1 = rk1 ^ rk0;
		rk2 = rk2 ^ rk1;
		rk3 = rk2 ^ rk3;

		rk[roundCount * 4 + 4] = rk0;
		rk[roundCount * 4 + 5] = rk1;
		rk[roundCount * 4 + 6] = rk2;
		rk[roundCount * 4 + 7] = rk3;
	}
}

// CTR encryption with one table extended as 32 columns
// 1 Table [256][32] -> arithmetic shift: __byte_perm function
// SBox[256] is partly expanded
__global__ void counterWithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBox(u32* pt, u32* rk, u32* t0G, u32* t4G, u64* range) {

	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int warpThreadIndex = threadIdx.x & 31;
	int warpThreadIndexSBox = warpThreadIndex % S_BOX_BANK_SIZE;

	// <SHARED MEMORY>
	__shared__ u32 t0S[TABLE_SIZE][SHARED_MEM_BANK_SIZE];
	__shared__ u32 t4S[TABLE_SIZE][S_BOX_BANK_SIZE];
	__shared__ u32 rkS[AES_128_KEY_SIZE_INT];

	if (threadIdx.x < TABLE_SIZE) {
		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) {
			t0S[threadIdx.x][bankIndex] = t0G[threadIdx.x];
		}

		for (u8 bankIndex = 0; bankIndex < S_BOX_BANK_SIZE; bankIndex++) {
			t4S[threadIdx.x][bankIndex] = t4G[threadIdx.x];
		}

		if (threadIdx.x < AES_128_KEY_SIZE_INT) {
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
		for (u8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1; roundCount++) {

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
		s0 = (t4S[t0 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t1 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t2 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t3) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[40];
		s1 = (t4S[t1 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t2 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t3 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t0) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[41];
		s2 = (t4S[t2 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t3 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t0 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t1) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[42];
		s3 = (t4S[t3 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t0 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t1 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t2) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[43];

		// Overflow
		if (pt3Init == MAX_U32) {
			pt2Init++;
		}

		pt3Init++;
		
	}

	if (threadIndex == 1048575) {
		printf("threadIndex : %d\n", threadIndex);
		printf("Plaintext   : %08x %08x %08x %08x\n", pt0Init, pt1Init, pt2Init, pt3Init);
		printf("Ciphertext  : %08x %08x %08x %08x\n", s0, s1, s2, s3);
		printf("-------------------------------\n");
	}
/*	pt[0] ^= s0;
	pt[1] ^= s0;
	pt[2] ^= s0;
	pt[3] ^= s0;*/
}
__global__ void counterWithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBoxCihangir(u32* pt, u32* rk, u32* t0G, u32* t4G, u64* range, u8* SAES) {
	u64 threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int warpThreadIndex = threadIdx.x & 31;
	__shared__ u32 t0S[TABLE_SIZE][SHARED_MEM_BANK_SIZE];
	__shared__ u8 Sbox[64][32][4];
//	__shared__ u32 t4S[TABLE_SIZE][S_BOX_BANK_SIZE];
	__shared__ u32 rkS[AES_128_KEY_SIZE_INT];
	if (threadIdx.x < TABLE_SIZE) {
		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) {	
			t0S[threadIdx.x][bankIndex] = t0G[threadIdx.x];
			Sbox[threadIdx.x / 4][bankIndex][threadIdx.x % 4] = SAES[threadIdx.x];
		}
//		for (u8 bankIndex = 0; bankIndex < S_BOX_BANK_SIZE; bankIndex++) {	t4S[threadIdx.x][bankIndex] = t4G[threadIdx.x];	}
//		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) { Sbox[threadIdx.x / 4][bankIndex][threadIdx.x % 4] = SAES[threadIdx.x]; }
		if (threadIdx.x < AES_128_KEY_SIZE_INT) {rkS[threadIdx.x] = rk[threadIdx.x];}
	}
	__syncthreads();
	u32 pt0Init, pt1Init, pt2Init, pt3Init;
	u32 s0, s1, s2, s3;
	pt0Init = pt[0];	pt1Init = pt[1];	pt2Init = pt[2];	pt3Init = pt[3];
	u64 threadRange = *range;
	u64 threadRangeStart = pt2Init;
	threadRangeStart = threadRangeStart << 32;
	threadRangeStart ^= pt3Init;
	threadRangeStart += threadIndex * threadRange;
	pt2Init = threadRangeStart >> 32;
	pt3Init = threadRangeStart & 0xFFFFFFFF;
	for (u32 rangeCount = 0; rangeCount < threadRange; rangeCount++) {
		// Create plaintext as 32 bit unsigned integers
		s0 = pt0Init;		s1 = pt1Init;		s2 = pt2Init;		s3 = pt3Init;
		// First round just XORs input with key.
		s0 = s0 ^ rkS[0];		s1 = s1 ^ rkS[1];		s2 = s2 ^ rkS[2];		s3 = s3 ^ rkS[3];
		u32 t0, t1, t2, t3;
		for (u8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1; roundCount++) {
			// Table based round function
			u32 rkStart = roundCount * 4 + 4;
			t0 = t0S[s0 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s3 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rkS[rkStart];
			t1 = t0S[s1 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s0 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rkS[rkStart + 1];
			t2 = t0S[s2 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s1 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rkS[rkStart + 2];
			t3 = t0S[s3 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s2 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rkS[rkStart + 3];
			s0 = t0;			s1 = t1;			s2 = t2;			s3 = t3;
		}
		// Calculate the last round key
		// Last round uses s-box directly and XORs to produce output.
/*		s0 = (t4S[t0 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t1 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t2 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t3) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[40];
		s1 = (t4S[t1 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t2 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t3 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t0) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[41];
		s2 = (t4S[t2 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t3 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t0 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t1) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[42];
		s3 = (t4S[t3 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t0 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t1 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t2) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[43];*/
		s0 = arithmeticRightShiftBytePerm((u32)Sbox[((t0 >> 24)) / 4][warpThreadIndex][((t0 >> 24)) % 4], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t1 >> 16) & 0xff) / 4][warpThreadIndex][((t1 >> 16)) % 4], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t2 >> 8) & 0xFF) / 4][warpThreadIndex][((t2 >> 8)) % 4], SHIFT_3_RIGHT) ^ ((u32)Sbox[((t3 & 0xFF) / 4)][warpThreadIndex][((t3 & 0xFF) % 4)]) ^ rkS[40];
		s1 = arithmeticRightShiftBytePerm((u32)Sbox[((t1 >> 24)) / 4][warpThreadIndex][((t1 >> 24)) % 4], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t2 >> 16) & 0xff) / 4][warpThreadIndex][((t2 >> 16)) % 4], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t3 >> 8) & 0xFF) / 4][warpThreadIndex][((t3 >> 8)) % 4], SHIFT_3_RIGHT) ^ ((u32)Sbox[((t0 & 0xFF) / 4)][warpThreadIndex][((t0 & 0xFF) % 4)]) ^ rkS[41];
		s2 = arithmeticRightShiftBytePerm((u32)Sbox[((t2 >> 24)) / 4][warpThreadIndex][((t2 >> 24)) % 4], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t3 >> 16) & 0xff) / 4][warpThreadIndex][((t3 >> 16)) % 4], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t0 >> 8) & 0xFF) / 4][warpThreadIndex][((t0 >> 8)) % 4], SHIFT_3_RIGHT) ^ ((u32)Sbox[((t1 & 0xFF) / 4)][warpThreadIndex][((t1 & 0xFF) % 4)]) ^ rkS[42];
		s3 = arithmeticRightShiftBytePerm((u32)Sbox[((t3 >> 24)) / 4][warpThreadIndex][((t3 >> 24)) % 4], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t0 >> 16) & 0xff) / 4][warpThreadIndex][((t0 >> 16)) % 4], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t1 >> 8) & 0xFF) / 4][warpThreadIndex][((t1 >> 8)) % 4], SHIFT_3_RIGHT) ^ ((u32)Sbox[((t2 & 0xFF) / 4)][warpThreadIndex][((t2 & 0xFF) % 4)]) ^ rkS[43];
		// Overflow
		if (pt3Init == MAX_U32) {			pt2Init++;		}
		pt3Init++;
	}
	if (threadIndex == 1048575) {
		printf("threadIndex : %d\n", threadIndex);
		printf("Plaintext   : %08x %08x %08x %08x\n", pt0Init, pt1Init, pt2Init, pt3Init);
		printf("Ciphertext  : %08x %08x %08x %08x\n", s0, s1, s2, s3);
		printf("-------------------------------\n");
	}
}
__global__ void counterWithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBoxCihangir2(u32* pt, u32* rk, u32* t0G, u32* t4G, u64* range, u8* SAES) {
	u64 threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int warpThreadIndex = threadIdx.x & 31;
	__shared__ u32 t0S[TABLE_SIZE][SHARED_MEM_BANK_SIZE];
	__shared__ u8 Sbox[64][32][4];
	//	__shared__ u32 t4S[TABLE_SIZE][S_BOX_BANK_SIZE];
	__shared__ u32 rkS[AES_128_KEY_SIZE_INT];
	if (threadIdx.x < TABLE_SIZE) {
		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) {
			t0S[threadIdx.x][bankIndex] = t0G[threadIdx.x];
			Sbox[threadIdx.x / 4][bankIndex][threadIdx.x % 4] = SAES[threadIdx.x];
		}
		//		for (u8 bankIndex = 0; bankIndex < S_BOX_BANK_SIZE; bankIndex++) {	t4S[threadIdx.x][bankIndex] = t4G[threadIdx.x];	}
		//		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) { Sbox[threadIdx.x / 4][bankIndex][threadIdx.x % 4] = SAES[threadIdx.x]; }
		if (threadIdx.x < AES_128_KEY_SIZE_INT) { rkS[threadIdx.x] = rk[threadIdx.x]; }
	}
	__syncthreads();
	u32 pt0Init, pt1Init, pt2Init, pt3Init;
	u32 s0, s1, s2, s3;
	pt0Init = pt[0];	pt1Init = pt[1];	pt2Init = pt[2];	pt3Init = pt[3];
	u64 threadRange = *range;
	u64 threadRangeStart = pt2Init;
	threadRangeStart = threadRangeStart << 32;
	threadRangeStart ^= pt3Init;
	threadRangeStart += threadIndex * threadRange;
	pt2Init = threadRangeStart >> 32;
	pt3Init = threadRangeStart & 0xFFFFFFFF;
	for (u32 rangeCount = 0; rangeCount < threadRange; rangeCount++) {
		// Create plaintext as 32 bit unsigned integers
		s0 = pt0Init;		s1 = pt1Init;		s2 = pt2Init;		s3 = pt3Init;
		// First round just XORs input with key.
		s0 = s0 ^ rkS[0];		s1 = s1 ^ rkS[1];		s2 = s2 ^ rkS[2];		s3 = s3 ^ rkS[3];
		u32 t0, t1, t2, t3;
		for (u8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1; roundCount++) {
			// Table based round function
			u32 rkStart = roundCount * 4 + 4;
			t0 = t0S[s0 >> 24][warpThreadIndex] ^ arithmeticRightShift(t0S[(s1 >> 16) & 0xFF][warpThreadIndex], 8) ^ arithmeticRightShift(t0S[(s2 >> 8) & 0xFF][warpThreadIndex], 16) ^ arithmeticRightShift(t0S[s3 & 0xFF][warpThreadIndex], 24) ^ rkS[rkStart];
			t1 = t0S[s1 >> 24][warpThreadIndex] ^ arithmeticRightShift(t0S[(s2 >> 16) & 0xFF][warpThreadIndex], 8) ^ arithmeticRightShift(t0S[(s3 >> 8) & 0xFF][warpThreadIndex], 16) ^ arithmeticRightShift(t0S[s0 & 0xFF][warpThreadIndex], 24) ^ rkS[rkStart + 1];
			t2 = t0S[s2 >> 24][warpThreadIndex] ^ arithmeticRightShift(t0S[(s3 >> 16) & 0xFF][warpThreadIndex], 8) ^ arithmeticRightShift(t0S[(s0 >> 8) & 0xFF][warpThreadIndex], 16) ^ arithmeticRightShift(t0S[s1 & 0xFF][warpThreadIndex], 24) ^ rkS[rkStart + 2];
			t3 = t0S[s3 >> 24][warpThreadIndex] ^ arithmeticRightShift(t0S[(s0 >> 16) & 0xFF][warpThreadIndex], 8) ^ arithmeticRightShift(t0S[(s1 >> 8) & 0xFF][warpThreadIndex], 16) ^ arithmeticRightShift(t0S[s2 & 0xFF][warpThreadIndex], 24) ^ rkS[rkStart + 3];
			s0 = t0;			s1 = t1;			s2 = t2;			s3 = t3;
		}
		// Calculate the last round key
		// Last round uses s-box directly and XORs to produce output.
/*		s0 = (t4S[t0 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t1 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t2 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t3) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[40];
		s1 = (t4S[t1 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t2 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t3 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t0) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[41];
		s2 = (t4S[t2 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t3 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t0 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t1) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[42];
		s3 = (t4S[t3 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t0 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t1 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t2) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[43];*/
		s0 = arithmeticRightShift((u64)Sbox[((t0 >> 24)) / 4][warpThreadIndex][((t0 >> 24)) % 4], 8) ^ arithmeticRightShift((u64)Sbox[((t1 >> 16) & 0xff) / 4][warpThreadIndex][((t1 >> 16)) % 4], 16) ^ arithmeticRightShift((u64)Sbox[((t2 >> 8) & 0xFF) / 4][warpThreadIndex][((t2 >> 8)) % 4], 24) ^ ((u64)Sbox[((t3 & 0xFF) / 4)][warpThreadIndex][((t3 & 0xFF) % 4)]) ^ rkS[40];
		s1 = arithmeticRightShift((u64)Sbox[((t1 >> 24)) / 4][warpThreadIndex][((t1 >> 24)) % 4], 8) ^ arithmeticRightShift((u64)Sbox[((t2 >> 16) & 0xff) / 4][warpThreadIndex][((t2 >> 16)) % 4], 16) ^ arithmeticRightShift((u64)Sbox[((t3 >> 8) & 0xFF) / 4][warpThreadIndex][((t3 >> 8)) % 4], 24) ^ ((u64)Sbox[((t0 & 0xFF) / 4)][warpThreadIndex][((t0 & 0xFF) % 4)]) ^ rkS[41];
		s2 = arithmeticRightShift((u64)Sbox[((t2 >> 24)) / 4][warpThreadIndex][((t2 >> 24)) % 4], 8) ^ arithmeticRightShift((u64)Sbox[((t3 >> 16) & 0xff) / 4][warpThreadIndex][((t3 >> 16)) % 4], 16) ^ arithmeticRightShift((u64)Sbox[((t0 >> 8) & 0xFF) / 4][warpThreadIndex][((t0 >> 8)) % 4], 24) ^ ((u64)Sbox[((t1 & 0xFF) / 4)][warpThreadIndex][((t1 & 0xFF) % 4)]) ^ rkS[42];
		s3 = arithmeticRightShift((u64)Sbox[((t3 >> 24)) / 4][warpThreadIndex][((t3 >> 24)) % 4], 8) ^ arithmeticRightShift((u64)Sbox[((t0 >> 16) & 0xff) / 4][warpThreadIndex][((t0 >> 16)) % 4], 16) ^ arithmeticRightShift((u64)Sbox[((t1 >> 8) & 0xFF) / 4][warpThreadIndex][((t1 >> 8)) % 4], 24) ^ ((u64)Sbox[((t2 & 0xFF) / 4)][warpThreadIndex][((t2 & 0xFF) % 4)]) ^ rkS[43];
		// Overflow
		if (pt3Init == MAX_U32) { pt2Init++; }
		pt3Init++;
	}
	if (threadIndex == 1048575) {
		printf("threadIndex : %I64d\n", threadIndex);
		printf("Plaintext   : %08x %08x %08x %08x\n", pt0Init, pt1Init, pt2Init, pt3Init);
		printf("Ciphertext  : %08x %08x %08x %08x\n", s0, s1, s2, s3);
		printf("-------------------------------\n");
	}
}
// CTR encryption with one table extended as 32 columns
// 1 Table [256][32] -> arithmetic shift: __byte_perm function
// 4 S-box, each shifted
__global__ void counterWithOneTableExtendedSharedMemoryBytePerm4ShiftedSbox(u32* pt, u32* rk, u32* t0G, u32* t4_0G, u32* t4_1G, u32* t4_2G, u32* t4_3G, u64* range) {

	u32 threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int warpThreadIndex = threadIdx.x & 31;

	// <SHARED MEMORY>
	__shared__ u32 t0S[TABLE_SIZE][SHARED_MEM_BANK_SIZE];
	__shared__ u32 t4_0S[TABLE_SIZE];
	__shared__ u32 t4_1S[TABLE_SIZE];
	__shared__ u32 t4_2S[TABLE_SIZE];
	__shared__ u32 t4_3S[TABLE_SIZE];
	__shared__ u32 rkS[AES_128_KEY_SIZE_INT];

	if (threadIdx.x < TABLE_SIZE) {
		t4_0S[threadIdx.x] = t4_0G[threadIdx.x];
		t4_1S[threadIdx.x] = t4_1G[threadIdx.x];
		t4_2S[threadIdx.x] = t4_2G[threadIdx.x];
		t4_3S[threadIdx.x] = t4_3G[threadIdx.x];
		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) {
			t0S[threadIdx.x][bankIndex] = t0G[threadIdx.x];
		}

		if (threadIdx.x < AES_128_KEY_SIZE_INT) {
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
		for (u8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1; roundCount++) {

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
		s0 = t4_3S[t0 >> 24] ^ t4_2S[(t1 >> 16) & 0xff] ^ t4_1S[(t2 >> 8) & 0xff] ^ t4_0S[(t3) & 0xFF] ^ rkS[40];
		s1 = t4_3S[t1 >> 24] ^ t4_2S[(t2 >> 16) & 0xff] ^ t4_1S[(t3 >> 8) & 0xff] ^ t4_0S[(t0) & 0xFF] ^ rkS[41];
		s2 = t4_3S[t2 >> 24] ^ t4_2S[(t3 >> 16) & 0xff] ^ t4_1S[(t0 >> 8) & 0xff] ^ t4_0S[(t1) & 0xFF] ^ rkS[42];
		s3 = t4_3S[t3 >> 24] ^ t4_2S[(t0 >> 16) & 0xff] ^ t4_1S[(t1 >> 8) & 0xff] ^ t4_0S[(t2) & 0xFF] ^ rkS[43];

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

__host__ int main128Ctr() {
	printf("\n");
	printf("########## AES-128 Counter Mode Implementation ##########\n");
	printf("\n");

	// Allocate plaintext and every round key
	u32 *pt, *rk, *roundKeys;
	gpuErrorCheck(cudaMallocManaged(&pt, 4 * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&rk, 4 * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&roundKeys, AES_128_KEY_SIZE_INT * sizeof(u32)));

	pt[0] = 0x3243F6A8U;
	pt[1] = 0x885A308DU;
	pt[2] = 0x313198A2U;
	pt[3] = 0x00000000U;

	rk[0] = 0x2B7E1516U;
	rk[1] = 0x28AED2A6U;
	rk[2] = 0xABF71588U;
	rk[3] = 0x09CF4F3CU;

	// Allocate RCON values
	u32* rcon;
	gpuErrorCheck(cudaMallocManaged(&rcon, RCON_SIZE * sizeof(u32)));
	for (int i = 0; i < RCON_SIZE; i++) {
		rcon[i] = RCON32[i];
	}

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
	for (int i = 0; i < 256; i++) SAES_d[i] = SAES[i]; // Cihangir
	printf("-------------------------------\n");
	u64* range = calculateRange();
/*	printf("Initial Plaintext              : %08x %08x %08x %08x\n", pt[0], pt[1], pt[2], pt[3]);
	printf("Initial Key                    : %08x %08x %08x %08x\n", rk[0], rk[1], rk[2], rk[3]);
	printf("-------------------------------\n");*/

	// Key expansion
	keyExpansion(rk, roundKeys);

	clock_t beginTime = clock();
	// Kernels
//	counterWithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBox<<<BLOCKS, THREADS>>>(pt, roundKeys, t0, t4, range);
	counterWithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBoxCihangir << <BLOCKS, THREADS >> > (pt, roundKeys, t0, t4, range, SAES_d);
//	counterWithOneTableExtendedSharedMemoryBytePerm4ShiftedSbox<<<BLOCKS, THREADS>>>(pt, roundKeys, t0, t4_0, t4_1, t4_2, t4_3, range);
//	cudaMemcpy(rk, pt, 4*sizeof(u32), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	printf("Time elapsed: %f sec\n", float(clock() - beginTime) / CLOCKS_PER_SEC);
	printf("-------------------------------\n");
	printLastCUDAError();
	printf("plaintext: %x %x %x %x\n",rk[0], rk[1], rk[2], rk[3]);

	// Free alocated arrays
	cudaFree(range);
	cudaFree(pt);
	cudaFree(rk);
	cudaFree(roundKeys);
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