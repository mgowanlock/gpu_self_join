# Distance similarity self-join 

Authors: Mike Gowanlock and Ben Karsin

Acknowledgements: This material is based upon work supported by the National Science Foundation under Grants 1849559, 1533823, and 1745331 and Fonds de la Recherche Scientifique-FNRS under Grant no MISU F 6001 1.

For more information, see the following papers:

* Gowanlock, M., & Karsin, B. (2019). Accelerating the similarity self-join using the GPU. ***Journal of parallel and distributed computing***, 133, 107-123.
* Gowanlock, M., & Karsin, B. (2019). GPU-Accelerated Similarity Self-Join for Multi-Dimensional Data. ***In Proceedings of the 15th International Workshop on Data Management on New Hardware*** (pp. 1-9).

## Overview

The self-join finds all points within a search distance ***epsilon*** of each other. This code is the basis of the publications outlined above. The code is suitable for ***exact*** distance similarity self-joins using the Euclidean distance. Additionally, the code can be used for low- and high-dimensional data. Descriptions of the algorithms for low- and high-dimensional data are given in the papers above.

Compiling and running the code:
* The code is written in CUDA, and C/C++.
* This code has been tested under Ubuntu with CUDA v9.0 on Nvidia GP100 and Titan X GPUs (both are Pascal generation GPUs).
* Modify the flags in the makefile for your architecture.

To run the code:
* ./main \<dataset filename\> \<epsilon\> \<data dimensionality\> \<algorithm number\>
* The GPU-accelerated self-join code uses algorithm number 3. The other algorithms are likely of little interest (they are CPU brute force, GPU brute force, and a sequential R-tree algorithm).
* For example "$./main test.txt 1.0 4 3" will compute the self-join on a 4-dimensional dataset stored in test.txt using a search distance of 1.0.
* The program outputs a file called gpu_stats.txt which prints the total execution time, the total result set size, the dataset file name and all of the parameters used.

## Overview: Python Interface
A Python interface has been provided that uses a shared library to access the C/CUDA code. This may allow you to skip reading all of the details below. But, if you're interested in the C/CUDA implementation, then you should skip reading this section.

To use the Python interface, edit the makefile with your settings and compile the shared library using the Makefile target as follows:
$make make_python_shared_lib

The settings that you will need to update are:
* The desired data dimensionality and number of indexed dimensions.
* The compute capability of your GPU.
* Any changes to the params.h file (described below). It is unlikely that you will need to change any parameters, although one common change might be to use doubles instead of floats.

After compiling the shared library, you can use the Python interface, where gdsjoingpu.py is Python wrapper around the C/CUDA code, and GDSJoin_test_example.py shows an example of using the library.

The interface returns the number of neighbors found for each object, and the neighbortable which is the list of which point ids are within epsilon of each point in the dataset.

Note that the neighborTable is stored as one 1-D contiguous array, so to obtain the list of neighbors for each point in Python, you need to use the numNeighbors array.

The interface is below:
numNeighbors, neighborTable = dssj.gdsjoin(dataset, epsilon, numdim, dtype, verbose) 

## Parameters

The file params.h specifies parameters and optimizations. Default values and categorization into low or high dimensionality are outlined below. The parameter names are in uppercase.

* GPUNUMDIM- Dimensionality of the data (n)
* NUMINDEXEDDIM- Number of dimensions indexed (k, default for high-D is 6-8 where k<n) 
* DTYPE- Data type of the data (float or double) 
* PRINTNEIGHBORTABLE- Prints the list of neighbors for each point (default 0)

The following are used to count metrics for various experimental evaluations and alternative algorithm designs (leave these as defaults, unless reproducing performance evaluation) 
* SEARCHFILTTIME (default 0)
* COUNTMETRICS (default 0)
* LINEARSTAMP 0 (default 0)
* CTYPE (default unsigned long long)

The following are used for the batch estimator.
* GPUBUFFERSIZE (default 100000000)
* GPUSTREAMS (default 3)
* MINBATCHES (default 3)
* SAMPLERATE (default 0.015)

Optimizations:
* STAMP (unicomp optimization, default 1) used for low dimensionality, not extensively tested on high dimensional data.

High dimensionality (likely do not want to enable these at low-dimensionality, but results may vary)
* SORT (sortidu optimization, default 0)
* REORDER (reorder data by variance, default 1)
* SHORTCIRCUIT (default 1)
* ILP (default 8)
* QUERYREORDER (default 1)
* Optimizations ILP, QUERYREORDER are in a paper under review 


## Testing

A small dataset has been included for testing purposes. If you run the 2-D dataset test_data.txt with epsilon=1.0, with no optimizations and store the data as single precision floats you should obtain a total of 27982028 neighbors. 

$./main test_dat.txt 1.0 2 3



