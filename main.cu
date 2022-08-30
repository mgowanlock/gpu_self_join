#include <pthread.h>
#include <cstdlib>
#include <stdio.h>
#include <random>
#include "omp.h"
#include <algorithm> 
#include <string.h>
#include <fstream>
#include <iostream>
#include <string>
#include "GPU.h"
#include "kernel.h"

#ifndef PYTHON
#include "tree_index.h"
#endif

#include <math.h>
#include <queue>
#include <iomanip>
#include <set>
#include <algorithm>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>


//for printing defines as strings
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

//sort descending
bool compareByDimVariance(const dim_reorder_sort &a, const dim_reorder_sort &b)
{
    return a.variance > b.variance;
}


using namespace std;

//function prototypes
uint64_t getLinearID_nDimensions(unsigned int * indexes, unsigned int * dimLen, unsigned int nDimensions);
void populateNDGridIndexAndLookupArray(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE epsilon, struct gridCellLookup ** gridCellLookupArr, struct grid ** index, unsigned int * indexLookupArr,  DTYPE* minArr, unsigned int * nCells, uint64_t totalCells, unsigned int * nNonEmptyCells, unsigned int ** gridCellNDMask, unsigned int * gridCellNDMaskOffsets, unsigned int * nNDMaskElems);
void generateNDGridDimensions(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE epsilon, DTYPE* minArr, DTYPE* maxArr, unsigned int * nCells, uint64_t * totalCells);
void importNDDataset(std::vector<std::vector <DTYPE> > *dataPoints, char * fname);
void CPUBruteForceTable(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE epsilon, table * neighborTable, unsigned int * totalNeighbors);
void sortInNDBins(std::vector<std::vector <DTYPE> > *dataPoints);
void ReorderByDimension(std::vector<std::vector <DTYPE> > *NDdataPoints);
void storeOutlierScoresForPython(unsigned int databaseSize, struct neighborTableLookup * neighborTable, unsigned int * totalNumberOfNeighbors, unsigned int * outlierScoreArr);

//printing of the neighbortable
void printNeighborTable(unsigned int databaseSize, struct neighborTableLookup * neighborTable);

//store the neighbors for python wrapper
unsigned int * storeNeighborTableContiguousPython(unsigned int databaseSize, struct neighborTableLookup * neighborTable);
//store the number of neighbors for the Python wrapper
void storeNumberOfNeighborsForPython(unsigned int databaseSize, struct neighborTableLookup * neighborTable, unsigned int * totalNumberOfNeighbors);

//sort ascending
bool compareByPointValue(const keyValNumPointsStruct &a, const keyValNumPointsStruct &b)
{
    return a.counts < b.counts;
}

#ifndef PYTHON //standard C version
int main(int argc, char *argv[])
{

	//check that the number of data dimensions is greater than or equal to the number of indexed dimensions
	assert(GPUNUMDIM>=NUMINDEXEDDIM);
		
	omp_set_max_active_levels(3);
	/////////////////////////
	// Get information from command line
	//1) the dataset, 2) epsilon, 3) number of dimensions
	/////////////////////////

	//Read in parameters from file:
	//dataset filename and cluster instance file
	if (argc!=5)
	{
	cout <<"\n\nIncorrect number of input parameters.  \nShould be dataset file, epsilon, number of dimensions, searchmode\n";
	return 0;
	}
	
	//copy parameters from commandline:
	//char inputFname[]="data/test_data_removed_nan.txt";	
	char inputFname[500];
	char inputEpsilon[500];
	char inputnumdim[500];

	strcpy(inputFname,argv[1]);
	strcpy(inputEpsilon,argv[2]);
	strcpy(inputnumdim,argv[3]);

        int SEARCHMODE = atoi(argv[4]);

	DTYPE epsilon=atof(inputEpsilon);
	unsigned int NDIM=atoi(inputnumdim);

	if (GPUNUMDIM!=NDIM){
		printf("\nERROR: The number of dimensions defined for the GPU is not the same as the number of dimensions\n \
		 passed into the computer program on the command line. GPUNUMDIM=%d, NDIM=%d Exiting!!!",GPUNUMDIM,NDIM);
		return 0;
	}

	printf("\nDataset file: %s",inputFname);
	printf("\nEpsilon: %f",epsilon);
	printf("\nNumber of dimensions (NDIM): %d\n",NDIM);

	//////////////////////////////
	//import the dataset:
	/////////////////////////////
	
	
	std::vector<std::vector <DTYPE> > NDdataPoints;
	importNDDataset(&NDdataPoints, inputFname);


	//CPU brute force
     if(SEARCHMODE == 0) {

	//neighbor table:
	table * neighborTable;
	neighborTable=new table[NDdataPoints.size()];
	unsigned int totalNeighbors=0;
	printf("\nBrute force CPU: ");

	CPUBruteForceTable(&NDdataPoints, epsilon, neighborTable, &totalNeighbors);
	printf("\nTotal neighbors in table: %u",totalNeighbors);

	//test output:

	for (int i=0; i<NDdataPoints.size(); i++){
		printf("\npoint id: %d, neighbors: ",neighborTable[i].pointID);
		for (int j=0; j<neighborTable[i].neighbors.size(); j++){
			printf("%d,",neighborTable[i].neighbors[j]);
		}
	}
	
	}


	

	///////////////
	//BRUTE FORCE GPU
	//NO BATCHING
	///////////////
	if(SEARCHMODE == 1) {


	double tstart_bruteforcegpu=omp_get_wtime();

	//neighbor table:
	table * neighborTable;
	neighborTable=new table[NDdataPoints.size()];
	unsigned long long int * totalNeighbors;
	totalNeighbors=(unsigned long long int*)malloc(sizeof(unsigned long long int));
	*totalNeighbors=0;

	printf("\nBrute force GPU (NO BATCHING):");
	

	double tstart=omp_get_wtime();
	makeDistanceTableGPUBruteForce(&NDdataPoints,&epsilon, neighborTable, totalNeighbors);
	double tend=omp_get_wtime();
	printf("\nBRUTE FORCE Time on GPU: %f",tend-tstart);cout.flush();
	printf("\nTotal neighbours in table: %llu", *totalNeighbors);

	printf("\n*********************************");
	
	}



	//GPU with Grid index
	if(SEARCHMODE == 3) {

	char fname[]="gpu_stats.txt";
	ofstream gpu_stats;
	gpu_stats.open(fname,ios::app);	

	printf("\n*****************\nWarming up GPU:\n*****************\n");
	warmUpGPU();
	printf("\n*****************\n");

	DTYPE * minArr= new DTYPE[NUMINDEXEDDIM];
	DTYPE * maxArr= new DTYPE[NUMINDEXEDDIM];
	unsigned int * nCells= new unsigned int[NUMINDEXEDDIM];
	uint64_t totalCells=0;
	unsigned int nNonEmptyCells=0;
	uint64_t totalNeighbors =0;
	double totalTime=0;
	double timeReorderByDimVariance=0;	

	#if REORDER==1
	double reorder_start=omp_get_wtime();
	ReorderByDimension(&NDdataPoints);
	double reorder_end=omp_get_wtime();
	timeReorderByDimVariance= reorder_end - reorder_start;
	#endif

	
	double tstart_index=omp_get_wtime();
	generateNDGridDimensions(&NDdataPoints,epsilon, minArr, maxArr, nCells, &totalCells);
	printf("\nGrid: total cells (including empty) %lu",totalCells);

		



	// allocate memory for index now that we know the number of cells
	//the grid struct itself
	//the grid lookup array that accompanys the grid -- so we only send the non-empty cells
	struct grid * index; //allocate in the populateDNGridIndexAndLookupArray -- only index the non-empty cells
	struct gridCellLookup * gridCellLookupArr; //allocate in the populateDNGridIndexAndLookupArray -- list of non-empty cells

	//the grid cell mask tells you what cells are non-empty in each dimension
	//used for finding the non-empty cells that you want
	unsigned int * gridCellNDMask; //allocate in the populateDNGridIndexAndLookupArray -- list of cells in each n-dimension that have elements in them
	unsigned int * nNDMaskElems= new unsigned int; //size of the above array
	unsigned int * gridCellNDMaskOffsets=new unsigned int [NUMINDEXEDDIM*2]; //offsets into the above array for each dimension
																	//as [min,max,min,max,min,max] (for 3-D)	

	//ids of the elements in the database that are found in each grid cell
	unsigned int * indexLookupArr=new unsigned int[NDdataPoints.size()]; 
	populateNDGridIndexAndLookupArray(&NDdataPoints, epsilon, &gridCellLookupArr, &index, indexLookupArr, minArr,  nCells, totalCells, &nNonEmptyCells, &gridCellNDMask, gridCellNDMaskOffsets, nNDMaskElems);
	// populateNDGridIndexAndLookupArrayParallel(&NDdataPoints, epsilon, &gridCellLookupArr, &index, indexLookupArr, minArr,  nCells, totalCells, &nNonEmptyCells, &gridCellNDMask, gridCellNDMaskOffsets, nNDMaskElems);
	double tend_index=omp_get_wtime();
	printf("\nTime to index (not counted in the time): %f", tend_index - tstart_index);
	

	//Neighbortable storage -- the result
	neighborTableLookup * neighborTable= new neighborTableLookup[NDdataPoints.size()];
	std::vector<struct neighborDataPtrs> pointersToNeighbors;

	CTYPE* workCounts = (CTYPE*)malloc(2*sizeof(CTYPE));
	workCounts[0]=0;
	workCounts[1]=0;



	pointersToNeighbors.clear();

	double tstart=omp_get_wtime();	

	distanceTableNDGridBatches(&NDdataPoints, &epsilon, index, gridCellLookupArr, &nNonEmptyCells,  minArr, nCells, indexLookupArr, neighborTable, &pointersToNeighbors, &totalNeighbors, gridCellNDMask, gridCellNDMaskOffsets, nNDMaskElems, workCounts);
	
	double tend=omp_get_wtime();

	printf("\nTime: %f",(tend-tstart)+timeReorderByDimVariance);

	totalTime+=(tend-tstart)+timeReorderByDimVariance;


#if COUNTMETRICS==1
	gpu_stats<<totalTime<<", "<< inputFname<<", "<<epsilon<<", "<<totalNeighbors<<", GPUNUMDIM/NUMINDEXEDDIM/ILP/STAMP/SORT/REORDER/SHORTCIRCUIT/QUERYREORDER/DTYPE(float/double): "<<GPUNUMDIM<<", "<<NUMINDEXEDDIM<<", "<<ILP<<", "<<STAMP<<", "<<SORT<<", "<<REORDER<< ", "<<SHORTCIRCUIT<<", "<<QUERYREORDER<<", "<<STR(DTYPE)<<", COMPS/CELLCOMPS: " << workCounts[0] << ", " << workCounts[1] << endl;
#else
	gpu_stats<<totalTime<<", "<< inputFname<<", "<<epsilon<<", "<<totalNeighbors<<", GPUNUMDIM/NUMINDEXEDDIM/ILP/STAMP/SORT/REORDER/SHORTCIRCUIT/QUERYREORDER/DTYPE(float/double): "<<GPUNUMDIM<<", "<<NUMINDEXEDDIM<<", "<<ILP<<", "<<STAMP<<", "<<SORT<<", "<<REORDER<< ", "<<SHORTCIRCUIT<<", "<<QUERYREORDER<<", "<<STR(DTYPE)<<endl;
#endif
	gpu_stats.close();

	//Print NeighborTable:

	//We print based on whether unicomp is on or off.
	//Some related neighbortable data are shown below.

	#if PRINTNEIGHBORTABLE==1
	#if STAMP==0
	printNeighborTable(NDdataPoints.size(), neighborTable);
	// for (int i=0; i<NDdataPoints.size(); i++){
	// 	// sort to compare against CPU implementation
	// 	std::sort(neighborTable[i].dataPtr+neighborTable[i].indexmin,neighborTable[i].dataPtr+neighborTable[i].indexmax+1);
	// 	printf("\npoint id: %d, neighbors: ",i);
	// 	for (int j=neighborTable[i].indexmin; j<=neighborTable[i].indexmax; j++){
	// 		printf("%d,",neighborTable[i].dataPtr[j]);
	// 	}
		
	// }

	
	// For printing number of neighbors per object (with coords):
	// printf("\npoint id (coords), num neighbors: ");
	// for (int i=0; i<NDdataPoints.size(); i++){
	// 		printf("\n(%f, %f, %f)  %d, %d,", NDdataPoints[i][0], NDdataPoints[i][1], NDdataPoints[i][2], i,neighborTable[i].indexmax-neighborTable[i].indexmin+1);
	// }

	// For printing number of neighbors per object (without coords):
	// printf("\npoint id (coords), num neighbors: ");
	// for (int i=0; i<NDdataPoints.size(); i++){
	// 		printf("\n%d, %d,",i,neighborTable[i].indexmax-neighborTable[i].indexmin+1);
	// }


	//Count number of neighbors using neighbortable:
	// unsigned long int countTotalNeighbors=0;
	// for (int i=0; i<NDdataPoints.size(); i++){
	// 		countTotalNeighbors+=(neighborTable[i].indexmax-neighborTable[i].indexmin)+1;
	// }

	// printf("\nTotal neighbors from neighbortable: %lu",countTotalNeighbors);
	#endif //endif stamp==0

	
	#if STAMP==1
	for (int i=0; i<NDdataPoints.size(); i++){

		printf("\npoint id: %d, neighbors: ",i);
		// printf("\npoint id: %d, cntNDataArrays: %d: ",i, neighborTable[i].cntNDataArrays);
		//used for sorting the neighbors to compare neighbortables for validation
		std::vector<int>tmp;
		for (int j=0; j<neighborTable[i].cntNDataArrays; j++)
		{
			for (int k=neighborTable[i].vectindexmin[j]; k<=neighborTable[i].vectindexmax[j]; k++)
			{
				tmp.push_back(neighborTable[i].vectdataPtr[j][k]);
			}
		}

		//print sorted vector
		std::sort(tmp.begin(), tmp.end());
		for (int l=0; l<tmp.size(); l++)
		{
			printf("%d,",tmp[l]);
		}	
	}
	#endif //end if stamp==1
	#endif //endif print neighbortable

}


//Sequential R-tree implementation
if(SEARCHMODE == 9) {

	char fname[]="sequential_stats.txt";
	ofstream sequential_stats;
	sequential_stats.open(fname,ios::app);	
	unsigned long int numNeighbors=0;


	double averageTime=RtreeSearch(&NDdataPoints, epsilon, &numNeighbors);
	
	sequential_stats<<averageTime<<", "<<inputFname<<", "<<epsilon<<", "<<numNeighbors<<endl;
	sequential_stats.close();
}


	printf("\n");
	return 0;
}
#endif //end #if not Python (standard C version)


#ifdef PYTHON
//this is a pointer used that will get used from the Python interface
//Needs to be global
unsigned int * neighborTableResultContiguous;
extern "C" void GDSJoinPy(DTYPE * dataset, unsigned int NUMPOINTS, DTYPE epsilon, unsigned int NDIM, int gpuContext, unsigned int * outNumNeighborsWithinEps)
{

	//check that the number of data dimensions is greater than or equal to the number of indexed dimensions
	assert(GPUNUMDIM>=NUMINDEXEDDIM);

	//check that STAMP is disabled, as storing the neighbortable for Python with STAMP is enabled was not implemented
	#if STAMP==1
	fprintf(stderr,"\nError: Note that storing the neighbortable was only implemented when unicomp is disabled (STAMP==0).\n")
	assert(STAMP==0);
	#endif
			
	if (GPUNUMDIM!=NDIM){
		printf("\nERROR: The number of dimensions defined for the GPU in the shared library is not the same as the number of dimensions defined in the Python interface.\n GPUNUMDIM=%d, NDIM=%d Exiting!!!",GPUNUMDIM,NDIM);
		return;
	}

	omp_set_max_active_levels(3);

	cudaSetDevice(gpuContext);

	printf("\nEpsilon: %f",epsilon);
	printf("\nNumber of dimensions (NDIM): %d\n",NDIM);

	//////////////////////////////
	//import the dataset:
	/////////////////////////////
	
	
	std::vector<std::vector <DTYPE> > NDdataPoints;

	//copy data into the dataset vector

  	for (unsigned int i=0; i<NUMPOINTS; i++){
  		unsigned int idxMin=i*GPUNUMDIM;
  		unsigned int idxMax=(i+1)*GPUNUMDIM;
		std::vector<DTYPE>tmpPoint(dataset+idxMin, dataset+idxMax);
		NDdataPoints.push_back(tmpPoint);
	}
	


	char fname[]="gpu_stats.txt";
	ofstream gpu_stats;
	gpu_stats.open(fname,ios::app);	

	// printf("\n*****************\nWarming up GPU:\n*****************\n");
	// warmUpGPU();
	// printf("\n*****************\n");

	DTYPE * minArr= new DTYPE[NUMINDEXEDDIM];
	DTYPE * maxArr= new DTYPE[NUMINDEXEDDIM];
	unsigned int * nCells= new unsigned int[NUMINDEXEDDIM];
	uint64_t totalCells=0;
	unsigned int nNonEmptyCells=0;
	uint64_t totalNeighbors =0;
	double totalTime=0;
	double timeReorderByDimVariance=0;	

	#if REORDER==1
	double reorder_start=omp_get_wtime();
	ReorderByDimension(&NDdataPoints);
	double reorder_end=omp_get_wtime();
	timeReorderByDimVariance= reorder_end - reorder_start;
	#endif

	
	double tstart_index=omp_get_wtime();
	generateNDGridDimensions(&NDdataPoints,epsilon, minArr, maxArr, nCells, &totalCells);
	printf("\nGrid: total cells (including empty) %lu",totalCells);

		



	// allocate memory for index now that we know the number of cells
	//the grid struct itself
	//the grid lookup array that accompanys the grid -- so we only send the non-empty cells
	struct grid * index; //allocate in the populateDNGridIndexAndLookupArray -- only index the non-empty cells
	struct gridCellLookup * gridCellLookupArr; //allocate in the populateDNGridIndexAndLookupArray -- list of non-empty cells

	//the grid cell mask tells you what cells are non-empty in each dimension
	//used for finding the non-empty cells that you want
	unsigned int * gridCellNDMask; //allocate in the populateDNGridIndexAndLookupArray -- list of cells in each n-dimension that have elements in them
	unsigned int * nNDMaskElems= new unsigned int; //size of the above array
	unsigned int * gridCellNDMaskOffsets=new unsigned int [NUMINDEXEDDIM*2]; //offsets into the above array for each dimension
																	//as [min,max,min,max,min,max] (for 3-D)	

	//ids of the elements in the database that are found in each grid cell
	unsigned int * indexLookupArr=new unsigned int[NDdataPoints.size()]; 
	populateNDGridIndexAndLookupArray(&NDdataPoints, epsilon, &gridCellLookupArr, &index, indexLookupArr, minArr,  nCells, totalCells, &nNonEmptyCells, &gridCellNDMask, gridCellNDMaskOffsets, nNDMaskElems);
	// populateNDGridIndexAndLookupArrayParallel(&NDdataPoints, epsilon, &gridCellLookupArr, &index, indexLookupArr, minArr,  nCells, totalCells, &nNonEmptyCells, &gridCellNDMask, gridCellNDMaskOffsets, nNDMaskElems);
	double tend_index=omp_get_wtime();
	printf("\nTime to index (not counted in the time): %f", tend_index - tstart_index);
	

	//Neighbortable storage -- the result
	neighborTableLookup * neighborTable= new neighborTableLookup[NDdataPoints.size()];
	std::vector<struct neighborDataPtrs> pointersToNeighbors;

	CTYPE* workCounts = (CTYPE*)malloc(2*sizeof(CTYPE));
	workCounts[0]=0;
	workCounts[1]=0;



	pointersToNeighbors.clear();

	double tstart=omp_get_wtime();	

	distanceTableNDGridBatches(&NDdataPoints, &epsilon, index, gridCellLookupArr, &nNonEmptyCells,  minArr, nCells, indexLookupArr, neighborTable, &pointersToNeighbors, &totalNeighbors, gridCellNDMask, gridCellNDMaskOffsets, nNDMaskElems, workCounts);
	
	double tend=omp_get_wtime();

	printf("\nTime: %f",(tend-tstart)+timeReorderByDimVariance);

	totalTime+=(tend-tstart)+timeReorderByDimVariance;


	//Print NeighborTable:

	//We print based on whether unicomp is on or off.
	//Some related neighbortable data are shown below.


	
	#if STAMP==0
	//pointer for the python implementation to all of the neighbors
	storeNumberOfNeighborsForPython(NDdataPoints.size(), neighborTable, outNumNeighborsWithinEps);
	neighborTableResultContiguous=storeNeighborTableContiguousPython(NDdataPoints.size(), neighborTable);
	#endif


	//Free neighbortable memory to prevent memory leak when using the shared library called by the Python library
	for (int i=0; i<pointersToNeighbors.size(); i++)
	{
		delete pointersToNeighbors[i].dataPtr;
	}
	
	




	
}

extern "C" void copyResultIntoPythonArray(unsigned int * outPythonNeighborTable, unsigned int numResults)
{
	std::copy(neighborTableResultContiguous, neighborTableResultContiguous+numResults, outPythonNeighborTable);
	free(neighborTableResultContiguous);
}
#endif //end #ifdef Python 

unsigned int * storeNeighborTableContiguousPython(unsigned int databaseSize, struct neighborTableLookup * neighborTable)
{
	//Determine total size of the result set
	unsigned long int resultSize=0;
	for (int i=0; i<databaseSize; i++){		
		resultSize+=neighborTable[i].indexmax-neighborTable[i].indexmin+1;
	}

	//allocate memory for contiguous array
	unsigned int * neighborTableResult = (unsigned int *)malloc(sizeof(unsigned int)*resultSize);

	//this is for validation
	// uint64_t totalCountNeighbors=0;
	

	unsigned long int cnt=0;
	for (int i=0; i<databaseSize; i++){
		for (int j=neighborTable[i].indexmin; j<=neighborTable[i].indexmax; j++){
			//store result for passing back to Python
			neighborTableResult[cnt]=neighborTable[i].dataPtr[j];
			//increment cnt
			cnt++;
		}
	}

	// printf("\nSum of indices of neighbors: %lu", totalCountNeighbors);
	return neighborTableResult;

}

void printNeighborTable(unsigned int databaseSize, struct neighborTableLookup * neighborTable)
{

	char fname[]="DSSJ_out.txt";
	ofstream DSSJ_out;
	DSSJ_out.open(fname,ios::out);	

	printf("\n\nOutputting neighbors to: %s\n", fname);
	DSSJ_out<<"#data point (line is the point id), neighbor point ids\n";

	for (int i=0; i<databaseSize; i++){
		//sort to have increasing point IDs
		std::sort(neighborTable[i].dataPtr+neighborTable[i].indexmin,neighborTable[i].dataPtr+neighborTable[i].indexmax+1);
		for (int j=neighborTable[i].indexmin; j<=neighborTable[i].indexmax; j++){
			DSSJ_out<<neighborTable[i].dataPtr[j]<<", ";
		}
		DSSJ_out<<"\n";
		
	}

	DSSJ_out.close();
}


void printOutlierScores(unsigned int databaseSize,struct neighborTableLookup * neighborTable)
{

	///////////////////
	//Outlier criterion: print the number of points each point has within epsilon


	//For each point, compute the total squared distances to its k neighbors
	unsigned int * totalNumberOfNeighbors = (unsigned int *)malloc(sizeof(unsigned int)*databaseSize);
	struct keyValNumPointsStruct * keyValuePairPointIDNumNeighbors = (struct keyValNumPointsStruct *)malloc(sizeof(keyValNumPointsStruct)*databaseSize);
	for (unsigned int i=0; i<databaseSize; i++)
	{
		totalNumberOfNeighbors[i]=neighborTable[i].indexmax-neighborTable[i].indexmin+1;
		
		//for sorting key/value pairs
		keyValuePairPointIDNumNeighbors[i].pointID=i;
		keyValuePairPointIDNumNeighbors[i].counts=totalNumberOfNeighbors[i];
	}

	//sort the point IDs and values by key/value pair
	std::sort(keyValuePairPointIDNumNeighbors, keyValuePairPointIDNumNeighbors+databaseSize, compareByPointValue);
	
	//store the scores for each point in an array that will be printed
	int * outlierScoreArr=(int *)malloc(sizeof(int)*databaseSize);

	for(int i=0; i<databaseSize; i++)
	{
		int pointId=keyValuePairPointIDNumNeighbors[i].pointID;
		outlierScoreArr[pointId]=i;
	}

	//end outlier criterion
	///////////////////


	//print to file for each point: its sum of distances to all points and its outlier ranking
	
	char fname[]="DSSJ_outlier_scores.txt";
	ofstream DSSJ_out;
	DSSJ_out.open(fname, ios::out);	

	printf("\nOutputting outlier scores to: %s\n", fname);
	DSSJ_out<<"#data point (line is the point id), col0: Number of neighbors the point has within epsilon, ";
	DSSJ_out<<"col1: outlier ranking for col0\n";

	for(int i=0; i<databaseSize; i++)
	{
		DSSJ_out<<totalNumberOfNeighbors[i]<<", "<<outlierScoreArr[i]<<endl;
	}

	DSSJ_out.close();

	//free all memory allocated in this function
	free(outlierScoreArr);
	free(totalNumberOfNeighbors);
	free(keyValuePairPointIDNumNeighbors);
	

}



void storeNumberOfNeighborsForPython(unsigned int databaseSize, struct neighborTableLookup * neighborTable, unsigned int * totalNumberOfNeighbors)
{
	for (unsigned int i=0; i<databaseSize; i++)
	{
		totalNumberOfNeighbors[i]=neighborTable[i].indexmax-neighborTable[i].indexmin+1;
	}
}


// void storeOutlierScoresForPython(unsigned int databaseSize, struct neighborTableLookup * neighborTable, unsigned int * totalNumberOfNeighbors, unsigned int * outlierScoreArr)
// {

// 	///////////////////
// 	//Outlier criterion: print the number of points each point has within epsilon



// 	//For each point, compute the total squared distances to its k neighbors
	
// 	struct keyValNumPointsStruct * keyValuePairPointIDNumNeighbors = (struct keyValNumPointsStruct *)malloc(sizeof(keyValNumPointsStruct)*databaseSize);
// 	for (unsigned int i=0; i<databaseSize; i++)
// 	{
// 		totalNumberOfNeighbors[i]=neighborTable[i].indexmax-neighborTable[i].indexmin+1;
		
// 		//for sorting key/value pairs
// 		keyValuePairPointIDNumNeighbors[i].pointID=i;
// 		keyValuePairPointIDNumNeighbors[i].counts=totalNumberOfNeighbors[i];
// 	}

// 	//sort the point IDs and values by key/value pair
// 	std::sort(keyValuePairPointIDNumNeighbors, keyValuePairPointIDNumNeighbors+databaseSize, compareByPointValue);
	
// 	//store the scores for each point in an array
	

// 	for(int i=0; i<databaseSize; i++)
// 	{
// 		int pointId=keyValuePairPointIDNumNeighbors[i].pointID;
// 		outlierScoreArr[pointId]=i;
// 	}

// 	//end outlier criterion
// 	///////////////////


// }



struct cmpStruct {
	cmpStruct(std::vector <std::vector <DTYPE>> points) {this -> points = points;}
	bool operator() (int a, int b) {
		return points[a][0] < points[b][0];
	}

	std::vector<std::vector<DTYPE>> points;
};



void populateNDGridIndexAndLookupArray(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE epsilon, struct gridCellLookup ** gridCellLookupArr, struct grid ** index, unsigned int * indexLookupArr,  DTYPE* minArr, unsigned int * nCells, uint64_t totalCells, unsigned int * nNonEmptyCells, unsigned int ** gridCellNDMask, unsigned int * gridCellNDMaskOffsets, unsigned int * nNDMaskElems)
{

	/////////////////////////////////
	//Populate grid lookup array
	//and corresponding indicies in the lookup array
	/////////////////////////////////
	printf("\n\n*****************************\nPopulating Grid Index and lookup array:\n*****************************\n");
	// printf("\nSize of dataset: %lu", NDdataPoints->size());


	///////////////////////////////
	//First, we need to figure out how many non-empty cells there will be
	//For memory allocation
	//Need to do a scan of the dataset and calculate this
	//Also need to keep track of the list of uniquie linear grid cell IDs for inserting into the grid
	///////////////////////////////
	std::set<uint64_t> uniqueGridCellLinearIds;
	std::vector<uint64_t>uniqueGridCellLinearIdsVect; //for random access

	for (int i=0; i<NDdataPoints->size(); i++){
		unsigned int tmpNDCellIdx[NUMINDEXEDDIM];
		for (int j=0; j<NUMINDEXEDDIM; j++){
			tmpNDCellIdx[j]=(((*NDdataPoints)[i][j]-minArr[j])/epsilon);
		}
		uint64_t linearID=getLinearID_nDimensions(tmpNDCellIdx, nCells, NUMINDEXEDDIM);
		uniqueGridCellLinearIds.insert(linearID);

	}

	// printf("uniqueGridCellLinearIds: %d",uniqueGridCellLinearIds.size());

	//copy the set to the vector (sets can't do binary searches -- no random access)
	std::copy(uniqueGridCellLinearIds.begin(), uniqueGridCellLinearIds.end(), std::back_inserter(uniqueGridCellLinearIdsVect));
	



	///////////////////////////////////////////////


	std::vector<uint64_t> * gridElemIDs;
	gridElemIDs = new std::vector<uint64_t>[uniqueGridCellLinearIds.size()];

	//Create ND array mask:
	//This mask determines which cells in each dimension has points in them.
	std::set<unsigned int> NDArrMask[NUMINDEXEDDIM];
	
	vector<uint64_t>::iterator lower;
	

	for (int i=0; i<NDdataPoints->size(); i++){
		unsigned int tmpNDCellID[NUMINDEXEDDIM];
		for (int j=0; j<NUMINDEXEDDIM; j++){
			tmpNDCellID[j]=(((*NDdataPoints)[i][j]-minArr[j])/epsilon);

			//add value to the ND array mask
			NDArrMask[j].insert(tmpNDCellID[j]);
		}

		//get the linear id of the cell
		uint64_t linearID=getLinearID_nDimensions(tmpNDCellID, nCells, NUMINDEXEDDIM);
		//printf("\nlinear id: %d",linearID);
		if (linearID > totalCells){

			printf("\n\nERROR Linear ID is: %lu, total cells is only: %lu\n\n", linearID, totalCells);
		}

		//find the index in gridElemIds that corresponds to this grid cell linear id
		
		lower=std::lower_bound(uniqueGridCellLinearIdsVect.begin(), uniqueGridCellLinearIdsVect.end(),linearID);
		uint64_t gridIdx=lower - uniqueGridCellLinearIdsVect.begin();
		gridElemIDs[gridIdx].push_back(i);
	}

	
	

	///////////////////////////////
	//Here we fill a temporary index with points, and then copy the non-empty cells to the actual index
	///////////////////////////////
	
	struct grid * tmpIndex=new grid[uniqueGridCellLinearIdsVect.size()];

	int cnt=0;

	

	//populate temp index and lookup array

	for (int i=0; i<uniqueGridCellLinearIdsVect.size(); i++)
	{
			tmpIndex[i].indexmin=cnt;
			for (int j=0; j<gridElemIDs[i].size(); j++)
			{
				if (j>((NDdataPoints->size()-1)))
				{
					printf("\n\n***ERROR Value of a data point is larger than the dataset! %d\n\n", j);
					return;
				}
				indexLookupArr[cnt]=gridElemIDs[i][j]; 
				cnt++;
			}
			tmpIndex[i].indexmax=cnt-1;
	}

	// printf("\nExiting grid populate method early!");
	// return;

	printf("\nFull cells: %d (%f, fraction full)",(unsigned int)uniqueGridCellLinearIdsVect.size(), uniqueGridCellLinearIdsVect.size()*1.0/double(totalCells));
	printf("\nEmpty cells: %ld (%f, fraction empty)",totalCells-(unsigned int)uniqueGridCellLinearIdsVect.size(), (totalCells-uniqueGridCellLinearIdsVect.size()*1.0)/double(totalCells));
	
	*nNonEmptyCells=uniqueGridCellLinearIdsVect.size();


	printf("\nSize of index that would be sent to GPU (GiB) -- (if full index sent), excluding the data lookup arr: %f", (double)sizeof(struct grid)*(totalCells)/(1024.0*1024.0*1024.0));
	printf("\nSize of compressed index to be sent to GPU (GiB) , excluding the data and grid lookup arr: %f", (double)sizeof(struct grid)*(uniqueGridCellLinearIdsVect.size()*1.0)/(1024.0*1024.0*1024.0));


	/////////////////////////////////////////
	//copy the tmp index into the actual index that only has the non-empty cells

	//allocate memory for the index that will be sent to the GPU
	*index=new grid[uniqueGridCellLinearIdsVect.size()];
	*gridCellLookupArr= new struct gridCellLookup[uniqueGridCellLinearIdsVect.size()];

	cmpStruct theStruct(*NDdataPoints);

	for (int i=0; i<uniqueGridCellLinearIdsVect.size(); i++){
			(*index)[i].indexmin=tmpIndex[i].indexmin;
			(*index)[i].indexmax=tmpIndex[i].indexmax;
			(*gridCellLookupArr)[i].idx=i;
			(*gridCellLookupArr)[i].gridLinearID=uniqueGridCellLinearIdsVect[i];
	}

	printf("\nWhen copying from entire index to compressed index: number of non-empty cells: %lu",uniqueGridCellLinearIdsVect.size());
		
	//copy NDArrMask from set to an array

	//find the total size and allocate the array
	
	unsigned int cntNDOffsets=0;
	unsigned int cntNonEmptyNDMask=0;
	for (int i=0; i<NUMINDEXEDDIM; i++){
		cntNonEmptyNDMask+=NDArrMask[i].size();
	}	
	*gridCellNDMask = new unsigned int[cntNonEmptyNDMask];
	
	*nNDMaskElems=cntNonEmptyNDMask;

	
	//copy the offsets to the array
	for (int i=0; i<NUMINDEXEDDIM; i++){
		//Min
		gridCellNDMaskOffsets[(i*2)]=cntNDOffsets;
		for (std::set<unsigned int>::iterator it=NDArrMask[i].begin(); it!=NDArrMask[i].end(); ++it){
    		(*gridCellNDMask)[cntNDOffsets]=*it;
    		cntNDOffsets++;
		}
		//max
		gridCellNDMaskOffsets[(i*2)+1]=cntNDOffsets-1;
	}
	
	delete [] tmpIndex;
		


} //end function populate grid index and lookup array



//determines the linearized ID for a point in n-dimensions
//indexes: the indexes in the ND array: e.g., arr[4][5][6]
//dimLen: the length of each array e.g., arr[10][10][10]
//nDimensions: the number of dimensions


uint64_t getLinearID_nDimensions(unsigned int * indexes, unsigned int * dimLen, unsigned int nDimensions) {
    // int i;
    // uint64_t offset = 0;
    // for( i = 0; i < nDimensions; i++ ) {
    //     offset += (uint64_t)pow(dimLen[i],i) * (uint64_t)indexes[nDimensions - (i + 1)];
    // }
    // return offset;

    uint64_t index = 0;
	uint64_t multiplier = 1;
	for (int i = 0; i<nDimensions; i++){
  	index += (uint64_t)indexes[i] * multiplier;
  	multiplier *= dimLen[i];
	}

	return index;
}



//min arr- the minimum value of the points in each dimensions - epsilon
//we can use this as an offset to calculate where points are located in the grid
//max arr- the maximum value of the points in each dimensions + epsilon 
//returns the time component of sorting the dimensions when SORT=1
void generateNDGridDimensions(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE epsilon, DTYPE* minArr, DTYPE* maxArr, unsigned int * nCells, uint64_t * totalCells)
{

	printf("\n\n*****************************\nGenerating grid dimensions.\n*****************************\n");

	printf("\nNumber of dimensions data: %d, Number of dimensions indexed: %d", GPUNUMDIM, NUMINDEXEDDIM);
	
	//make the min/max values for each grid dimension the first data element
	for (int j=0; j<NUMINDEXEDDIM; j++){
		minArr[j]=(*NDdataPoints)[0][j];
		maxArr[j]=(*NDdataPoints)[0][j];
	}



	for (int i=1; i<NDdataPoints->size(); i++)
	{
		for (int j=0; j<NUMINDEXEDDIM; j++){
		if ((*NDdataPoints)[i][j]<minArr[j]){
			minArr[j]=(*NDdataPoints)[i][j];
		}
		if ((*NDdataPoints)[i][j]>maxArr[j]){
			maxArr[j]=(*NDdataPoints)[i][j];
		}	
		}
	}	
		

	printf("\n");
	for (int j=0; j<NUMINDEXEDDIM; j++){
		printf("Data Dim: %d, min/max: %f,%f\n",j,minArr[j],maxArr[j]);
	}	

	//add buffer around each dim so no weirdness later with putting data into cells
	for (int j=0; j<NUMINDEXEDDIM; j++){
		minArr[j]-=epsilon;
		maxArr[j]+=epsilon;
	}	

	for (int j=0; j<NUMINDEXEDDIM; j++){
		printf("Appended by epsilon Dim: %d, min/max: %f,%f\n",j,minArr[j],maxArr[j]);
	}	
	
	//calculate the number of cells:
	for (int j=0; j<NUMINDEXEDDIM; j++){
		nCells[j]=ceil((maxArr[j]-minArr[j])/epsilon);
		printf("Number of cells dim: %d: %d\n",j,nCells[j]);
	}

	//calc total cells: num cells in each dim multiplied
	uint64_t tmpTotalCells=nCells[0];
	for (int j=1; j<NUMINDEXEDDIM; j++){
		tmpTotalCells*=nCells[j];
	}

	*totalCells=tmpTotalCells;

}



//CPU brute force
void CPUBruteForceTable(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE epsilon, table * neighborTable, unsigned int * totalNeighbors)
{
	DTYPE runningDist=0;
	unsigned int runningNeighbors=0;
	for (int i=0; i<NDdataPoints->size(); i++)
	{
		neighborTable[i].pointID=i;
		for (int j=0; j<NDdataPoints->size(); j++)
		{
			runningDist=0;
			for (int k=0; k<GPUNUMDIM; k++){
				runningDist+=((*NDdataPoints)[i][k]-(*NDdataPoints)[j][k])*((*NDdataPoints)[i][k]-(*NDdataPoints)[j][k]);
			}
			
			//if within epsilon:
			if ((sqrt(runningDist))<=epsilon){
				neighborTable[i].neighbors.push_back(j);
				runningNeighbors++;
			}
		}

	}
	//update the total neighbor count
	(*totalNeighbors)=runningNeighbors;

}

//reorders the input data by variance of each dimension
void ReorderByDimension(std::vector<std::vector <DTYPE> > *NDdataPoints)
{
	
	double tstart_sort=omp_get_wtime();
	DTYPE sums[GPUNUMDIM];
	DTYPE average[GPUNUMDIM];
	struct dim_reorder_sort dim_variance[GPUNUMDIM];
	for (int i=0; i< GPUNUMDIM; i++){
		sums[i]=0;
		average[i]=0;
	}

	DTYPE greatest_variance=0;
	int greatest_variance_dim=0;

	
	int sample=100;
	DTYPE inv_sample=1.0/(sample*1.0);
	printf("\nCalculating variance based on on the following fraction of pts: %f",inv_sample);
	double tvariancestart=omp_get_wtime();
		//calculate the variance in each dimension	
		for (int i=0; i<GPUNUMDIM; i++)
		{
			//first calculate the average in the dimension:
			//only use every 10th point
			for (int j=0; j<(*NDdataPoints).size(); j+=sample)
			{
			sums[i]+=(*NDdataPoints)[j][i];
			}


			average[i]=(sums[i])/((*NDdataPoints).size()*inv_sample);
			// printf("\nAverage in dim: %d, %f",i,average[i]);

			//Next calculate the std. deviation
			sums[i]=0; //reuse this for other sums
			for (int j=0; j<(*NDdataPoints).size(); j+=sample)
			{
			sums[i]+=(((*NDdataPoints)[j][i])-average[i])*(((*NDdataPoints)[j][i])-average[i]);
			}
			
			dim_variance[i].variance=sums[i]/((*NDdataPoints).size()*inv_sample);
			dim_variance[i].dim=i;
			
			// printf("\nDim:%d, variance: %f",dim_variance[i].dim,dim_variance[i].variance);

			if(greatest_variance<dim_variance[i].variance)
			{
				greatest_variance=dim_variance[i].variance;
				greatest_variance_dim=i;
			}
		}


	// double tvarianceend=omp_get_wtime();
	// printf("\nTime to compute variance only: %f",tvarianceend - tvariancestart);
	//sort based on variance in dimension:

	// double tstartsortreorder=omp_get_wtime();
	std::sort(dim_variance,dim_variance+GPUNUMDIM,compareByDimVariance); 	

	for (int i=0; i<GPUNUMDIM; i++)
	{
		printf("\nReodering dimension by: dim: %d, variance: %f",dim_variance[i].dim,dim_variance[i].variance);
	}

	printf("\nDimension with greatest variance: %d",greatest_variance_dim);

	//copy the database
	// double * tmp_database= (double *)malloc(sizeof(double)*(*NDdataPoints).size()*(GPUNUMDIM));  
	// std::copy(database, database+((*DBSIZE)*(GPUNUMDIM)),tmp_database);
	std::vector<std::vector <DTYPE> > tmp_database;

	//copy data into temp database
	tmp_database=(*NDdataPoints);

	
	
	#pragma omp parallel for num_threads(5) shared(NDdataPoints, tmp_database)
	for (int j=0; j<GPUNUMDIM; j++){

		int originDim=dim_variance[j].dim;	
		for (int i=0; i<(*NDdataPoints).size(); i++)
		{	
			(*NDdataPoints)[i][j]=tmp_database[i][originDim];
		}
	}

	double tend_sort=omp_get_wtime();
	// double tendsortreorder=omp_get_wtime();
	// printf("\nTime to sort/reorder only: %f",tendsortreorder-tstartsortreorder);
	double timecomponent=tend_sort - tstart_sort;
	printf("\nTime to reorder cols by variance (this gets added to the time because its an optimization): %f",timecomponent);
	
}

