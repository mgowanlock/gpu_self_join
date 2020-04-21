#include "structs.h"
#include "params.h"

//original with unsigned long long for the counter (atomic)

//with long long unsigned int
__global__ void kernelBruteForce(unsigned int *N, unsigned int *debug1, unsigned int *debug2, DTYPE *epsilon, 
	unsigned long long int * cnt, DTYPE* database, int * pointIDKey, int * pointInDistVal);

__global__ void kernelNDGridIndexBatchEstimator(unsigned int *debug1, unsigned int *debug2, unsigned int *N,  
	unsigned int * sampleOffset, DTYPE* database, DTYPE* epsilon, struct grid * index, unsigned int * indexLookupArr, 
	struct gridCellLookup * gridCellLookupArr, DTYPE* minArr, unsigned int * nCells, unsigned int * cnt, 
	unsigned int * nNonEmptyCells,  unsigned int * gridCellNDMask, unsigned int * gridCellNDMaskOffsets, unsigned int * orderedQueryPntIDs);

__global__ void kernelNDGridIndexGlobal(unsigned int *debug1, unsigned int *debug2, unsigned int *N,  
	unsigned int * offset, unsigned int *batchNum, DTYPE * database, DTYPE *epsilon, struct grid * index, unsigned int * indexLookupArr, 
	struct gridCellLookup * gridCellLookupArr, DTYPE* minArr, unsigned int * nCells, unsigned int * cnt, 
	unsigned int * nNonEmptyCells,  unsigned int * gridCellNDMask, unsigned int * gridCellNDMaskOffsets,
	int * pointIDKey, int * pointInDistVal, unsigned int * orderedQueryPntIDs, CTYPE* workCounts);

__device__ uint64_t getLinearID_nDimensionsGPU(unsigned int * indexes, unsigned int * dimLen, unsigned int nDimensions);

__global__ void kernelSortPointsInCells(DTYPE* database, struct grid * index, unsigned int* indexLookupArr, unsigned int nNonEmptyCells);

__global__ void kernelUniqueKeys(int * pointIDKey, unsigned int * N, int * uniqueKey, int * uniqueKeyPosition, unsigned int * cnt);

__device__ void evaluateCell(unsigned int* nCells, unsigned int* indexes, struct gridCellLookup * gridCellLookupArr, unsigned int* nNonEmptyCells, DTYPE* database, DTYPE* epsilon, struct grid * index, unsigned int * indexLookupArr, DTYPE* point, unsigned int* cnt,int* pointIDKey, int* pointInDistVal, int pointIdx, bool differentCell, unsigned int* nDCellIDs, CTYPE* workCounts);