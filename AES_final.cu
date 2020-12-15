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
#include "AES_final.h"
//
#include "128-es.cuh"
#include "128-ctr.cuh"
#include "192-es.cuh"
#include "192-ctr.cuh"
#include "256-es.cuh"
#include "256-ctr.cuh"
//#include "small.cuh"
//#include "silent.cuh"
#include "file-encryption.cuh"

void selection(int choice) {
	if (choice == 1) main128ExhaustiveSearch(1);
	else if (choice == 2) main128ExhaustiveSearch(2);
	else if (choice == 3) main128Ctr();
	else if (choice == 4) main192ExhaustiveSearch();
	else if (choice == 5) main192Ctr();
	else if (choice == 6) main256ExhaustiveSearch();
	else if (choice == 7) main256Ctr();
	else if (choice == 8) {
		main128ExhaustiveSearch(1);
		main128Ctr();
		main192ExhaustiveSearch();
		main192Ctr();
		main256ExhaustiveSearch();
		main256Ctr();
	}
	else printf("Wrong selection\n");
}

int main() {
	cudaSetDevice(0);
	int choice;
	printf(
		"(1) AES-128 Exhaustive Search (no bank conflict)\n"
		"(2) AES-128 Exhaustive Search (conflicting S-box)\n"
		"(3) AES-128 CTR \n"
		"(4) AES-192 Exhaustive Search\n"
		"(5) AES-192 CTR\n"
		"(6) AES-256 Exhaustive Search\n"
		"(7) AES-256 CTR\n"
		"(8) ALL\n"
		"Choice: ");
	scanf_s("%d", &choice);
	selection(choice);
//  AES-128 Exhaustive Search
//	main128ExhaustiveSearch();

	// AES-128 Counter Mode
//	main128Ctr();

	// AES-192 Exhaustive Search
//	main192ExhaustiveSearch();


	// AES-192 Counter Mode
//	main192Ctr();

	// AES-256 Exhaustive Search
//	main256ExhaustiveSearch();

	// AES-256 Counter Mode
//	main256Ctr();

	// Small AES probability calculation
	//mainSmall();

	// Silent
	//mainSilent();

	// File Encryption
	//mainFileEncryption();
	return 0;
}
