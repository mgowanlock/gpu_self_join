#include <vector>
#include "params.h"


//MBBS
struct Rect
{
	Rect()  {}
	  DTYPE Point[GPUNUMDIM];//point
	  DTYPE MBB_min[GPUNUMDIM]; //MBB min
	  DTYPE MBB_max[GPUNUMDIM]; //MBB max
	  int pid; //point id

  	void CreateMBB(){
		for (int i=0; i<GPUNUMDIM; i++){
			MBB_min[i]=Point[i];
			MBB_max[i]=Point[i];
		}	
	}
};


//neighbortable CPU -- indexmin and indexmax point to a single vector
struct neighborTableLookupCPU
{
	int pointID;
	int indexmin;
	int indexmax;
};

void createEntryMBBs(std::vector<std::vector <DTYPE> > *NDdataPoints, Rect * dataRects);
bool DBSCANmySearchCallbackSequential(int id, void* arg);
double RtreeSearch(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE epsilon, unsigned long int * numNeighbors);
void generateQueryMBB(std::vector<std::vector<DTYPE> > *NDdataPoints, unsigned int idx, DTYPE epsilon, DTYPE * MBB_min, DTYPE * MBB_max);
unsigned int filterCandidatesAddToTable(std::vector<std::vector<DTYPE> > *NDdataPoints, unsigned int idx, DTYPE epsilon, std::vector<unsigned int> * neighborList, neighborTableLookupCPU * neighborTable, std::vector<unsigned int > * neighborTableVect);

