# cuda_matmul
Place for optimized CUDA kernel used for ML

my_best_kernel in this repo demonstrates the techniques I used to achieve 95% the speed of the cublas sgemm.
Using my_best_kernel as a starting point, one can create their own kernels that have similar algorithms.

my_best_kernel is small and demonstrates some useful techniques to get the most speed from your kernels.  Like overcoming shared
memory bottlenecks by loading values into registers and using that registers around 4-8 times.  Aligning data reads from main memory
linear by thread id, aligning shared memory reads to either broadcast or not have cache collisions.  Etc.  It is around 45-ish lines
of code.  I've tuned this one for the Pascal chipset.
