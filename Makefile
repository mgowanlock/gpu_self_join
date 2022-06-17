
#see params.h for the parameters




SOURCES = main.cu GPU.cu kernel.cu import_dataset.cpp tree_index.cpp 
OBJECTS = import_dataset.o tree_index.o 
CUDAOBJECTS = GPU.o kernel.o main.o
CC = nvcc
EXECUTABLE = main

FLAGS = -std=c++14 -O3 -Xcompiler -fopenmp -arch=compute_60 -code=sm_60 -lcuda -lineinfo 
CFLAGS = -c -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES


all: $(EXECUTABLE)


import_dataset.o: import_dataset.cpp params.h
	$(CC) $(CFLAGS) $(FLAGS) import_dataset.cpp

tree_index.o: tree_index.cpp params.h
	$(CC) $(CFLAGS) $(FLAGS) tree_index.cpp	



main.o: main.cu params.h
	$(CC) $(FLAGS) $(CFLAGS) main.cu 

kernel.o: kernel.cu params.h
	$(CC) $(FLAGS) $(CFLAGS) kernel.cu 		

GPU.o: GPU.cu params.h
	$(CC) $(FLAGS) $(CFLAGS) GPU.cu 	



$(EXECUTABLE): $(OBJECTS) $(CUDAOBJECTS)
	$(CC) $(FLAGS) $^ -o $@




clean:
	rm $(OBJECTS)
	rm $(CUDAOBJECTS)
	rm main




