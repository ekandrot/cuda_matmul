all : matmul.app

matmul.app : matmul.cu
#	clang++ -O3 -std=c++14 --cuda-gpu-arch=sm_61 -Xarch_sm_61 -O3 -o $@ $< -L/usr/local/cuda/lib64 -ldl -lrt -lcudart -lcublas -lblas -I /usr/local/cuda/include
	nvcc --ptxas-options=-v -arch=sm_61 -O3 -g -lineinfo -o $@ $< -lcublas -lblas -I /usr/local/cuda/include
	
