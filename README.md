# cuda_matmul
Place for optimized CUDA kernel used for ML

I was starting another company a few years ago, this one focused on ML, and I needed a slightly different version of matmul than what cublas
supplied.  So I wrote one.  The version in this repo demonstrates the techniques I used to achieve 95% the speed of the cublas sgemm.
Using my_best_kernel as a starting point, one can create their own kernels that have similar algorithms.

Why use it?  The NVIDIA cublas kernels are highly optimized and are very useful.  But if you want to do something similar, but
not exactly the same, you'd have to start from scratch.  I searched and found many matmuls, from NVIDIA blogs and Universities -
they were either very complex or much slower (one half to one tenth the speed) or were wrong. So use this as your starting point,
I'm giving it away, free to use or change as will.

my_best_kernel is small and demonstrates some useful techniques to get the most speed from your kernels.  Like overcoming shared
memory bottlenecks by loading values into registers and using that registers around 4-8 times.  Aligning data reads from main memory
linear by thread id, aligning shared memory reads to either broadcast or not have cache collisions.  Etc.  It is around 45-ish lines
of code.  I've tuned this one for the Pascal chipset, becauses it is what I have at the time.  Change the compile options if you have
a different chipset, which I can not verify so I didn't want to push something I couldn't check.

I've cleaned it up a bit from when I was developing it, but kept some things to show the path it took.  Sizes are hard coded, etc,
so that it shows the techniques and can be used by you for other matrix shapes.

Hope this helps and is useful to you.

Edward Kandrot
