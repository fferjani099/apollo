CC=g++
CFLAGS=-O3 -fPIC
CUDAINCLUDE=/usr/local/cuda/include
CUDALIB=/usr/local/cuda/lib64

libgpu_intercept.so: gpu_intercept.cpp
	$(CC) $(CFLAGS) -shared gpu_intercept.cpp -I$(CUDAINCLUDE) -L$(CUDALIB) -lcudart -ldl -lpthread -o libgpu_intercept.so

all: libgpu_intercept.so

clean:
	rm -f libgpu_intercept.so
