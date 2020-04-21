#include "structs.h"
#include "params.h"



void makeDistanceTableGPUBruteForce(std::vector<std::vector <DTYPE> > * NDdataPoints, DTYPE* epsilon, struct table * neighborTable, unsigned long long int * totalNeighbors);

void distanceTableNDGridBatches(std::vector<std::vector<DTYPE> > * NDdataPoints, DTYPE* epsilon, struct grid * index, 
	struct gridCellLookup * gridCellLookupArr, unsigned int * nNonEmptyCells, DTYPE* minArr, unsigned int * nCells, 
	unsigned int * indexLookupArr, struct neighborTableLookup * neighborTable, std::vector<struct neighborDataPtrs> * pointersToNeighbors, 
	uint64_t * totalNeighbors, unsigned int * gridCellNDMask, unsigned int * gridCellNDMaskOffsets, unsigned int * nNDMaskElems, CTYPE* workCounts);


unsigned long long callGPUBatchEst(unsigned int * DBSIZE, DTYPE* dev_database, DTYPE* dev_epsilon, struct grid * dev_grid, 
	unsigned int * dev_indexLookupArr, struct gridCellLookup * dev_gridCellLookupArr, DTYPE* dev_minArr, 
	unsigned int * dev_nCells, unsigned int * dev_nNonEmptyCells, unsigned int * dev_gridCellNDMask, 
	unsigned int * dev_gridCellNDMaskOffsets, unsigned int * dev_nNDMaskElems, unsigned int * dev_orderedQueryPntIDs, unsigned int * retNumBatches, unsigned int * retGPUBufferSize);

void constructNeighborTableKeyValueWithPtrs(int * pointIDKey, int * pointInDistValue, struct neighborTableLookup * neighborTable, int * pointersToNeighbors, unsigned int * cnt);

void warmUpGPU();

//for the brute force version without batches
void constructNeighborTableKeyValue(int * pointIDKey, int * pointInDistValue, struct table * neighborTable, unsigned int * cnt);




//Unicomp requires multiple updates to the neighbortable for a given point
//This allows updating the neighbortable for the same point
void constructNeighborTableKeyValueWithPtrsWithMultipleUpdates(int * pointIDKey, int * pointInDistValue, struct neighborTableLookup * neighborTable, int * pointersToNeighbors, unsigned int * cnt, pthread_mutex_t * pointLocks);

//Unicomp requires multiple updates to the neighbortable for a given point
//This allows updating the neighbortable for the same point
//WITHOUT VECTORS FOR DATA
void constructNeighborTableKeyValueWithPtrsWithMultipleUpdatesMultipleDataArrays(int * pointIDKey, int * pointInDistValue, struct neighborTableLookup * neighborTable, int * pointersToNeighbors, unsigned int * cnt, int * uniqueKeys, int * uniqueKeyPosition, unsigned int numUniqueKeys);

//Unicomp requires multiple updates to the neighbortable for a given point
//This allows updating the neighbortable for the same point
//Without locks
void constructNeighborTableKeyValueWithPtrsBatchMaskArray(int * pointIDKey, int * pointInDistValue, struct neighborTableLookup * neighborTable, int * pointersToNeighbors, unsigned int * cnt, int batchNum);

//Sort the queries by their workload based on the number of points in the cell
//From hybrid KNN paper in GPGPU'19 
void computeWorkDifficulty(unsigned int * outputOrderedQueryPntIDs, struct gridCellLookup * gridCellLookupArr, unsigned int * nNonEmptyCells, unsigned int * indexLookupArr, struct grid * index);
