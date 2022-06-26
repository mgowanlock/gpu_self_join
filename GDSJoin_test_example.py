import csv
import numpy as np
import pandas as pd
#load the GDS-Join Python library
import  gdsjoingpu as dssj


def getColumn(filename, column):
    results = csv.reader(open(filename), delimiter=",")
    # next(results, None)  # skip the headers
    return [result[column] for result in results]




if __name__ == "__main__":

    #parameters
    
    fname="test_dat.txt"
    df = pd.read_csv(fname, delimiter=',', header=None)

    #flatten data to a list which is required by the library
    dataset = df.values.flatten().tolist()
    epsilon = 1.0
    
    verbose=False #True/False --- this is the C output from the shared library
    dtype="float"
    numdim=2
    

    numNeighbors, neighborTable, outlierRanking = dssj.gdsjoin(dataset, epsilon, numdim, dtype, verbose)    

        

    #this is only used so that we can run multiple examples at once and surpress the C stdout
    if(verbose==False):
        dssj.redirect_stdout()

