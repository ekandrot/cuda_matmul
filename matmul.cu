#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cooperative_groups.h>
using namespace cooperative_groups;

#include "cblas.h"
#include <chrono>
using namespace std::chrono;


#define LOOPAGE 100


static cublasStatus_t _cublaserr(cublasStatus_t err, const char*fname, int line_num) {
    const char* dummyStr = "unknown cublas error";
    if (err != CUBLAS_STATUS_SUCCESS) {
        printf("*** cublas error %d at line %d in %s:  %s\n", err, line_num, fname, dummyStr);
    }

    return err;
}

static cudaError_t _cuerr(cudaError_t err, const char*fname, int line_num) {
    if (err != cudaSuccess) {
        printf("*** cuda error %d at line %d in %s:  %s\n", err, line_num, fname, cudaGetErrorString(err));
    }

    return err;
}

#define cublaserr(err) _cublaserr(err, __FILE__, __LINE__)
#define cuerr(err) _cuerr(err, __FILE__, __LINE__)


void print_cublas_version(cublasHandle_t cublas_handle) {
    int version;
    cublaserr(cublasGetVersion(cublas_handle, &version));
    printf("cublas version:  %d\n", version);

    int value;
    cublaserr(cublasGetProperty(MAJOR_VERSION, &value));
    printf("cublas Major version:  %d", value);
    cublaserr(cublasGetProperty(MINOR_VERSION, &value));
    printf("  Minor version:  %d", value);
    cublaserr(cublasGetProperty(PATCH_LEVEL, &value));
    printf("  Patch level:  %d\n", value);

}

__device__ int sumReduction(thread_group g, float *x, int val) 
{ 
    // rank of this thread in the group 
    const int lane = g.thread_rank(); 

    // for each iteration of this loop, the number of threads active in the
    // reduction, i, is halved, and each active thread (with index [lane])
    // performs a single summation of it's own value with that
    // of a "partner" (with index [lane+i]). 
    for (int i = g.size()/2; i > 0; i /= 2) { 
        // store value for this thread in temporary array
        x[lane] = val;
        // synchronize all threads in group
        g.sync();
        if (lane<i) {
            // active threads perform summation of their value with
            // their partner's value
            val += x[lane + i];
        }
        // synchronize all threads in group
        g.sync();
    }

    // master thread in group returns result, and others return -1.
    if (lane==0)
        return val;
    else
        return -1;
}



#define MAX_THREADS_PER_BLOCK 256
#define MIN_BLOCKS_PER_MP     1

// does the same thing as kernel_53T, but internally handling 32 inner-index at a time, rather than 8
// still outputs 256x32 per block, just changes how it accumulates.
// runs slower on GP102 - 3.7 TFlops
__launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
__global__ void kernel_32(const float *a, const float *b, float *c) {
    const int tx = threadIdx.x;
    const int gr = blockIdx.x;
    const int gc = blockIdx.y;

    const int row = tx + gr * 256;
    const int col = 32 * gc;

    thread_block g = this_thread_block();

    __shared__ float block[32][32+1];

    float sum[32] = {0};

    for (int j=0; j<4096; j+=32) {
        float value[32];
        for (int k=0; k<32; ++k) {
            value[k] = a[row + (j + k)*4096];
        }

        // copy 32x32 block of B into shared memory
        int minor_index = tx & 31;  // range 0..31 (rows of B)
        int major_index = tx / 32;  // range 0..7  (cols of B)
        block[major_index][minor_index] = __ldg(b + (major_index + col) * 4096 + j+minor_index);
        block[major_index+8][minor_index] = __ldg(b + (major_index + 8 + col) * 4096 + j+minor_index);
        block[major_index+16][minor_index] = __ldg(b + (major_index + 16 + col) * 4096 + j+minor_index);
        block[major_index+24][minor_index] = __ldg(b + (major_index + 24 + col) * 4096 + j+minor_index);
        g.sync();

        // calc the partial 32-deep sum of the 32 return values this thread is handling
        for (int k=0; k<32; ++k) {
            for (int i=0; i<32; ++i) {
                sum[i] += value[k] * block[i][k];
            }
        }

        g.sync();
    }

    for (int j=0; j<32; ++j) {
        c[row + 4096*(j + col)] = sum[j];
    }
}




// dim3 blocks(16,128);
// dim3 threads(256);
// each block generates an output of 256x32
// runs around 5.3 TFlops/s on 4096x4096 * 4096x4096
__global__ void kernel_53T_old(const float *a, const float *b, float *c) {
    const int tx = threadIdx.x;
    const int gr = blockIdx.x;
    const int gc = blockIdx.y;

    const int row = tx + gr * 256;
    const int col = 32 * gc;

    thread_block g = this_thread_block();

    __shared__ float block[8][32];

    float sum[32] = {0};

    // each thread is calc'ing 32 values (ie each thread output is 1x32)
    // the middle index range is 4096, it takes that 8 at a time
    for (int j=0; j<4096; j+=8) {
        float value[8];
        for (int k=0; k<8; ++k) {
            value[k] = a[row + (j + k)*4096];
        }

        // each thread loads one value via __ldg (a read-only cache load op), then shares
        // reading linear in minor_index, so cuda can cache-read (will take 32 cache reads to fill shared memory this way)
        int minor_index = tx / 32;  // range 0..7  (rows of B)
        int major_index = tx & 31;  // range 0..31  (cols of B)
        block[minor_index][major_index] = __ldg(b + (major_index + col) * 4096 + j+minor_index);
        g.sync();

        // block [k][i] is thread independent, therefore a constant-broadcast from shared memory
        // each thread is calc'ing 8-deep partial sums, across the 32 cols it is going to return
        for (int k=0; k<8; ++k) {
            for (int i=0; i<32; ++i) {
                sum[i] += value[k] * block[k][i];
            }
        }

        g.sync();
    }

    for (int j=0; j<32; ++j) {
        c[row + 4096*(j + col)] = sum[j];
    }
}


// dim3 blocks(16,128);
// dim3 threads(256);
// each block generates an output of 256x32
// runs around 4.02 TFlops/s on 4096x4096 * 4096x4096
__global__ void kernel_53T(const float *a, const float *b, float *c) {
    const int tx = threadIdx.x;
    const int gr = blockIdx.x;
    const int gc = blockIdx.y;

    const int row = tx + gr * 256;
    const int col = 32 * gc;

    thread_block g = this_thread_block();

    __shared__ float block[8][32];

    float sum[32] = {0};

    float value0[8];
    float value1[8];
    // pre-load first 1x8 per thread
    for (int k=0; k<8; ++k) {
        value0[k] = a[row + k * 4096];
    }
    bool load_zero{false};

    // each thread is calc'ing 32 values (ie each thread output is 1x32)
    // the middle index range is 4096, it takes that 8 at a time
    for (int j=0; j<4096; j+=8) {
        float *value, *load_value;

        // pre-load 1x8 for next loop, if needed
        if (load_zero) {
            value = value1;
            load_value = value0;
        } else {
            value = value0;
            load_value = value1;
        }
        if (j+8 < 4096) {
            for (int k=0; k<8; ++k) {
                load_value[k] = a[row + (8+j + k)*4096];
            }
        }
        load_zero = !load_zero;

        // each thread loads one value via __ldg (a read-only cache load op), then shares
        // reading linear in minor_index, so cuda can cache-read (will take 32 cache reads to fill shared memory this way)
        int minor_index = tx / 32;  // range 0..7  (rows of B)
        int major_index = tx & 31;  // range 0..31  (cols of B)
        block[minor_index][major_index] = __ldg(b + (major_index + col) * 4096 + j+minor_index);
        g.sync();

        // block [k][i] is thread independent, therefore a constant-broadcast from shared memory
        // each thread is calc'ing 8-deep partial sums, across the 32 cols it is going to return
        for (int k=0; k<8; ++k) {
            for (int i=0; i<32; ++i) {
                sum[i] += value[k] * block[k][i];
            }
        }

        g.sync();
    }

    for (int j=0; j<32; ++j) {
        c[row + 4096*(j + col)] = sum[j];
    }
}


__global__ void kernel8x8(const float *A, const float *B, float *C) {
    const thread_block g = this_thread_block();
    const int tx = g.thread_index().x;
    const int ty = g.thread_index().y;
    const int gx = g.group_index().x;
    const int gy = g.group_index().y;

    // test out something with thread_groups here
    // if (gx == 0 && gy == 0) {
    //     thread_group tile4 = tiled_partition(g, 4);
    //     printf("%d, ", g.thread_rank() - tile4.thread_rank());
    //     tile4.sync();
    // }

    // starting point for the 8x8 output block owned by this thread
    // const int tr = 4*(tx + gx * 8);
    // const int tc = 4*(ty + gy * 8);

    __shared__ float buffer_A[32][32];
    __shared__ float buffer_B[32][32];

    float sum[4][4]={0};

    const int index = tx+ty*8;
    const int index31 = index & 31;
    const int index_div_32 = index / 32;

    for (int loop=0; loop<4096; loop+=32) {
        for (int i=0; i<32; i+=2) {
            buffer_A[i + index_div_32][index31] = A[(i + index_div_32 + loop)*4096 + gx*32 + index31];
            buffer_B[i + index_div_32][index31] = B[(i + index_div_32 + gy*32)*4096 + loop + index31];
        }
        g.sync();

        for (int j=0; j<32; ++j) {
            float row[4], col[4];
            for (int i=0; i<4; ++i) {
                row[i] = buffer_A[j][i + tx * 4];
                col[i] = buffer_B[i + ty * 4][j];
            }

            for (int c=0; c<4; ++c) {
                for (int r=0; r<4; ++r) {
                    sum[c][r] += row[r] * col[c];
                }
            }
        }
        g.sync();
    }

    // reuse an old shared buffer to get write speeds

    // for (int r=0; r<4; ++r) {
        for (int c=0; c<4; ++c) {
            float4 *sum4 = (float4*)(sum[c]);
            float4 *b4 = (float4*)(&buffer_A[c+ty*4][tx*4]);
            *b4 = *sum4;
            // buffer_A[c+ty*4][r+tx*4] = sum[c][r];
        }
    // }
    g.sync();

    for (int i=0; i<32; i+=2) {
        C[(i + index_div_32 + gy*32)*4096 + gx*32 + index31] = buffer_A[i + index_div_32][index31];
    }
}


// each thread owns 8x8 of C, which gives 8x2 reads and 64 fmas
// called with 256 threads, t(16,16) to leverage shared memory and 32 wide reads
// each block owns 128x128 of C, (16x8, 16x8)  16x16 - threads, each thread owning 8x8 output values
// there are 32x32 blocks - so this kernel is tuned for 4096x4906  (8 values x 16 threads x 32 blocks)
// first fills shared memory, using the 256 threads to move data from main memory - 2x128 16 times for A, 8x32 16 times for B
//    these reads are aligned so that threads 0..31 read from contiguous memory, as do threads 32..63, etc 
// The speed up trick comes from the idea that each thread doesn't need to own a contiguous range, but instead
//    should own cells so that reads/writes across the block are contiguous - thread 0 reads address zero, thread 1 reads adrress 1, etc.
//    rather than thread 0 owns 0..7, it will own 0, 16, 32, etc.  Thread 1 will own 1, 17, 33, etc.
//    so that thread.x is 128 wide, 8 threads skipping by 16, vs 16 groups of 8 threads that are linear

// around 10 Tflops on a GP102, which is about 95% the speed of cublas

// __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
__launch_bounds__(256, 2)
__global__ void real_kernel8x8(const float *A, const float *B, float *C) {
    const thread_block g = this_thread_block();
    const int tr = g.thread_index().x;      // 0..15
    const int tc = g.thread_index().y;      // 0..15
    const int gr = g.group_index().x * 128;       // 0..31  * 128
    const int gc = g.group_index().y * 128;       // 0..31  * 128

    // [c][r] for the indexes
    __shared__ float buffer_A[32][128];
    __shared__ float buffer_B[128][32];

    float sum[8][8]={};

    const int index = g.thread_rank();  // 0..255
    const int index127 = index & 127;
    const int index_div_128 = index / 128;
    const int index31 = index & 31;
    const int index_div_32 = index / 32;

    // this is the inner loop, which is the 32 wide read
    #pragma unroll
    for (int loop=0; loop<4096; loop+=32) {
        // use the rank of the thread to load our two caches, linearly
#if 0
        for (int i=0; i<16; ++i) {
            buffer_A[i*2 + index_div_128][index127] = A[(i*2 + index_div_128 + loop)*4096 + gr + index127];
            buffer_B[i*8 + index_div_32][index31] = B[(i*8 + index_div_32 + gc)*4096 + loop + index31];
        }
#else
        for (int i=0; i<32; i+=2) {
            buffer_A[i + index_div_128][index127] = A[(i + index_div_128 + loop)*4096 + gr + index127];
        }
        for (int i=0; i<128; i+=8) {
            buffer_B[i + index_div_32][index31] = B[(i + index_div_32 + gc)*4096 + loop + index31];
        }
#endif
        g.sync();

        // 32 for the inner 32 wide read
        for (int j=0; j<32; ++j) {
            // 16 groups of 8 = 128
            float row[8], col[8];
            for (int i=0; i<8; ++i) {
                row[i] = buffer_A[j][i*16 + tr];
                col[i] = buffer_B[i*16 + tc][j];
            }

            for (int c=0; c<8; ++c) {
                for (int r=0; r<8; ++r) {
                    sum[r][c] += row[r] * col[c];       // 1)  was sum[c][r] - swapping rc at here and 2, seems to be faster
                }
            }
        }
        g.sync();
    }

#if 1
    for (int c=0; c<8; ++c) {
        for (int r=0; r<8; ++r) {
            C[(gc + tc + c*16) * 4096 + (gr + tr + r*16)] = sum[r][c];  // 2) was sum[c][r] - swapping rc at here and 1, seems to be faster
        }
    }
#endif

#if 0
    for (int c=0; c<8; ++c) {
        // for (int r=0; r<8; +r) {
            *(float4 *)&(C[(gc + tc + c) * 4096 + (gr + tr)]) = *(float4 *)(sum[c]);
            *(float4 *)&(C[(gc + tc + c) * 4096 + (gr + tr + 4)]) = *(float4 *)&(sum[c][4]);
                    // }
    }
#endif


#if 0
    // each thread has 64 values, in an 8x8 grid.  the block has 128x128 values.
    // what is the best way to get those into shared memory than dumped?
    // or is just writing it to main memory directly okay?

    // reuse an old shared buffer to get write speeds

    // for (int r=0; r<4; ++r) {
        for (int c=0; c<4; ++c) {
            float4 *sum4 = (float4*)(sum[c]);
            float4 *b4 = (float4*)(&buffer_A[c+ty*4][tx*4]);
            *b4 = *sum4;
            // buffer_A[c+ty*4][r+tx*4] = sum[c][r];
        }
    // }
    g.sync();

    for (int i=0; i<32; i+=2) {
        C[(i + index_div_32 + gy*32)*4096 + gx*32 + index31] = buffer_A[i + index_div_32][index31];
    }
#endif
}


//
// based on real_kernel8x8 - it is the above code, minus all of the #if'ed out stuff, that was used along the way to write that kernel
//

__launch_bounds__(256, 2)
__global__ void my_best_kernel(const float *A, const float *B, float *C) {
    const thread_block g = this_thread_block();
    const int tr = g.thread_index().x;      // 0..15
    const int tc = g.thread_index().y;      // 0..15
    const int gr = g.group_index().x * 128;       // 0..31  * 128
    const int gc = g.group_index().y * 128;       // 0..31  * 128

    // [c][r] for the indexes
    __shared__ float buffer_A[32][128];
    __shared__ float buffer_B[128][32];

    float sum[8][8]={};

    const int index = g.thread_rank();  // 0..255
    const int index127 = index & 127;
    const int index_div_128 = index / 128;
    const int index31 = index & 31;
    const int index_div_32 = index / 32;

    // this is the inner loop, which is the 32 wide read
    #pragma unroll
    for (int loop=0; loop<4096; loop+=32) {
        // use the rank of the thread to load our two caches, linearly
        for (int i=0; i<32; i+=2) {
            buffer_A[i + index_div_128][index127] = A[(i + index_div_128 + loop)*4096 + gr + index127];
        }
        for (int i=0; i<128; i+=8) {
            buffer_B[i + index_div_32][index31] = B[(i + index_div_32 + gc)*4096 + loop + index31];
        }
        g.sync();

        // 32 for the inner 32 wide read
        for (int j=0; j<32; ++j) {
            // 16 groups of 8 = 128
            float row[8], col[8];
            for (int i=0; i<8; ++i) {
                row[i] = buffer_A[j][i*16 + tr];
                col[i] = buffer_B[i*16 + tc][j];
            }

            for (int c=0; c<8; ++c) {
                for (int r=0; r<8; ++r) {
                    sum[r][c] += row[r] * col[c];       // 1)  was sum[c][r] - swapping rc at here and 2, seems to be faster
                }
            }
        }
        g.sync();
    }

    for (int c=0; c<8; ++c) {
        for (int r=0; r<8; ++r) {
            C[(gc + tc + c*16) * 4096 + (gr + tr + r*16)] = sum[r][c];  // 2) was sum[c][r] - swapping rc at here and 1, seems to be faster
        }
    }
}


#define TILE_WIDTH 16
// From David Kirk - though it is in C format row-major, not Fortran col-major
__global__ void MatrixMulKernel(const float* d_M, const float* d_N, float* d_P, int Width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Identify the row and column of the d_P element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;
    // Loop over the d_M and d_N tiles required to compute d_P element
    for (int m = 0; m < Width/TILE_WIDTH; ++m) {
    // Coolaborative loading of d_M and d_N tiles into shared memory
        Mds[ty][tx] = d_M[Row*Width + m*TILE_WIDTH + tx];
        Nds[ty][tx] = d_N[(m*TILE_WIDTH + ty)*Width + Col];
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    d_P[Row*Width + Col] = Pvalue;
}


// dim3 blocks(256,256);
// dim3 threads(16,16);
// runs around 1.2 TFlops/s on 4096x4096 * 4096x4096
__global__ void kernel_1_2T(const float *a, const float *b, float *c) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gr = blockIdx.x;
    int gc = blockIdx.y;

    const int col = gc*16+ty;
    const int row = gr*16+tx;

    thread_block g = this_thread_block();

    __shared__ float ablk[16][16], bblk[16][16];

    float sum = 0;

    for (int i=0; i<4096; i+=16) {
        ablk[ty][tx] = a[row+(i+ty)*4096];
        bblk[ty][tx] = b[col*4096 + (i+tx)];
        g.sync();

        for (int j=0; j<16; ++j) {
            sum += ablk[j][tx] * bblk[ty][j];
        }
        g.sync();
    }
    c[row + 4096*col] = sum;
}


//----------------------------------------------------------------------------------------------------
// code from nvidia docs


// Thread block size
#define BLOCK_SIZE 16

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;
    int height;
    int stride; 
    float* elements;
} Matrix;

// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col,
                           float value)
{
    A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
 __device__ Matrix GetSubMatrix(Matrix A, int row, int col) 
{
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                         + BLOCK_SIZE * col];
    return Asub;
}

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(float *A, float *B, float *C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = d_A.stride = 4096; d_A.height = 4096;
    d_A.elements = A;

    Matrix d_B;
    d_B.width = d_B.stride = 4096; d_B.height = 4096;
    d_B.elements = B;

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = d_C.stride = 4096; d_C.height = 4096;
    d_C.elements = C;

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(4096 / dimBlock.x, 4096 / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
}

// Matrix multiplication kernel called by MatMul()
 __global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {

        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);

        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);

        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
}

//----------------------------------------------------------------------------------------------------





void validate_math(const float *validate, const float *c) {
    int incorrect=0;
    for (int i=0; i<4096*4096; ++i) {
        if (c[i] != c[i] || abs(validate[i] - c[i]) > 0.01) {
            ++incorrect;
            if (incorrect < 10) {
                printf("%d:  %f  %f\n", i, c[i], validate[i]);
            }
        }
    }
    printf("incorrect values %d\n", incorrect);
}


#if 0
__global__ void one_fma(float *f, long long int *cycles) {
    float _f = *f;
    volatile __shared__ float s_f;
    __syncthreads();
    long long int start = clock64();
    #pragma unroll
    for (int i=0; i<1000000; ++i) {
        // _f += _f * _f;
        _f = s_f;
        _f += _f * _f;
        _f += _f * _f;
        _f += _f * _f;
        _f += _f * _f;
        _f += _f * _f;
        _f += _f * _f;
        _f += _f * _f;
        _f += _f * _f;
        s_f = _f;
    }
    long long int end = clock64();
    *cycles = end - start;
    *f = s_f;
}

void time_one_fma() {
    long long int *cycles;
    cudaMallocManaged(&cycles, sizeof(long long int));
    float *f;
    cudaMallocManaged(&f, sizeof(float)); 
    *f = 1.0f;
    one_fma<<<28*16,128>>>(f,cycles);
    cudaDeviceSynchronize();
    printf("cycles per op = %lf\n", *cycles/(double)1e6);
}
#endif

int main() {
    // time_one_fma();

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0); // 0-th device
    printf("number of SMs:  %d\n", deviceProp.multiProcessorCount);
    

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    cublasHandle_t cublas_handle;

    cublaserr(cublasCreate(&cublas_handle));

    print_cublas_version(cublas_handle);

    const int MATRIX_SIZE = sizeof(float)*4096*4096;
    float *a, *b, *c, *validate;
    a = (float*)malloc(MATRIX_SIZE);
    b = (float*)malloc(MATRIX_SIZE);
    c = (float*)malloc(MATRIX_SIZE);
    validate = (float*)malloc(MATRIX_SIZE);
    for (int r=0; r<4096; ++r) {
        float v = std::rand();
        for (int col=0; col<4096; ++col) {
            int i = 4096*r+col;
            a[i] = std::rand();
            b[i] = std::rand();
            c[i] = 1.1;
            validate[i] = 1e32;
        }
    }

    float *da, *db, *dc, *dvalidate;
    cuerr(cudaMalloc(&da, MATRIX_SIZE));
    cuerr(cudaMalloc(&db, MATRIX_SIZE));
    cuerr(cudaMalloc(&dc, MATRIX_SIZE));
    cuerr(cudaMalloc(&dvalidate, MATRIX_SIZE));

    cudaEventRecord(start);
    cublaserr(cublasSetMatrix(4096, 4096, sizeof(float), a, 4096, da, 4096));
    cublaserr(cublasSetMatrix(4096, 4096, sizeof(float), b, 4096, db, 4096));
    cublaserr(cublasSetMatrix(4096, 4096, sizeof(float), c, 4096, dc, 4096));
    cublaserr(cublasSetMatrix(4096, 4096, sizeof(float), c, 4096, dvalidate, 4096));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float seconds = milliseconds / 1000;
    // printf("Memcpy time taken:  %0.3f ms\n", milliseconds);
    // printf("   %0.3f GB/sec\n", sizeof(float)*4096.0*4096*4/seconds/1e9);

    const float alpha = 1.0f;
    const float beta = 0;
    cudaEventRecord(start);
    for (int i=0; i<LOOPAGE; ++i) {
        cublaserr(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 4096, 4096, 4096,
            &alpha,
            da, 4096,
            db, 4096,
            &beta,
            dvalidate, 4096 ));
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    milliseconds /= LOOPAGE;
    printf("cublas Compute time taken:  %0.3f ms\n", milliseconds);
    seconds = milliseconds / 1000;
    printf("   %0.3f TFlops\n", 2.0*4096*4096*4096/seconds/1e12);
    cublaserr(cublasGetMatrix(4096, 4096, sizeof(float), dvalidate, 4096, validate, 4096));
    printf("\n");


    dim3 my_best_blocks8x8(32,32);
    dim3 my_best_threads8x8(16,16);

    // time my_best_kernel - currently based on real_kernel8x8
    cudaEventRecord(start);
    for (int i=0; i<LOOPAGE; ++i) {
        my_best_kernel<<<my_best_blocks8x8,my_best_threads8x8>>>(da, db, dc);
    }
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    milliseconds /= LOOPAGE;
    printf("my_best_kernel Compute time taken:  %0.3f ms\n", milliseconds);
    seconds = milliseconds / 1000;
    printf("   %0.3f TFlops\n", 2.0*4096*4096*4096/seconds/1e12);

    cublaserr(cublasGetMatrix(4096, 4096, sizeof(float), dc, 4096, c, 4096));
    validate_math(validate, c);
    cublaserr(cublasSetMatrix(4096, 4096, sizeof(float), c, 4096, dc, 4096));
    printf("\n");


#if 0
    dim3 r_blocks8x8(32,32);
    dim3 r_threads8x8(16,16);

    // time real_kernel8x8
    cudaEventRecord(start);
    for (int i=0; i<LOOPAGE; ++i) {
        real_kernel8x8<<<r_blocks8x8,r_threads8x8>>>(da, db, dc);
    }
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    milliseconds /= LOOPAGE;
    printf("real_kernel8x8 Compute time taken:  %0.3f ms\n", milliseconds);
    seconds = milliseconds / 1000;
    printf("   %0.3f TFlops\n", 2.0*4096*4096*4096/seconds/1e12);

    cublaserr(cublasGetMatrix(4096, 4096, sizeof(float), dc, 4096, c, 4096));
    validate_math(validate, c);
    cublaserr(cublasSetMatrix(4096, 4096, sizeof(float), c, 4096, dc, 4096));
    printf("\n");





    dim3 blocks8x8(4096/32,4096/32);
    dim3 threads8x8(8,8);

    // time kernel8x8
    cudaEventRecord(start);
    kernel8x8<<<blocks8x8,threads8x8>>>(da, db, dc);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("kernel8x8 Compute time taken:  %0.3f ms\n", milliseconds);
    seconds = milliseconds / 1000;
    printf("   %0.3f TFlops\n", 2.0*4096*4096*4096/seconds/1e12);

    cublaserr(cublasGetMatrix(4096, 4096, sizeof(float), dc, 4096, c, 4096));
    validate_math(validate, c);
    cublaserr(cublasSetMatrix(4096, 4096, sizeof(float), c, 4096, dc, 4096));
    printf("\n");


    // time Kirk's algo
    dim3 blocks16x16(4096/TILE_WIDTH, 4096/TILE_WIDTH);
    dim3 threads16x16(TILE_WIDTH,TILE_WIDTH);
    cublaserr(cublasSetMatrix(4096, 4096, sizeof(float), c, 4096, dc, 4096));

    cudaEventRecord(start);
    //MatrixMulKernel<<<blocks16x16,threads16x16>>>(da, db, dc, 4096);
    kernel_1_2T<<<blocks16x16,threads16x16>>>(da, db, dc);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("David Kirk's simple algo Compute time taken:  %0.3f ms\n", milliseconds);
    seconds = milliseconds / 1000;
    printf("   %0.3f TFlops\n", 2.0*4096*4096*4096/seconds/1e12);

    cublaserr(cublasGetMatrix(4096, 4096, sizeof(float), dc, 4096, c, 4096));
    validate_math(validate, c);
    cublaserr(cublasSetMatrix(4096, 4096, sizeof(float), c, 4096, dc, 4096));
    printf("\n");

    


    dim3 blocks(16,128);
    dim3 threads(256);
    cublaserr(cublasSetMatrix(4096, 4096, sizeof(float), c, 4096, dc, 4096));

    // time kernel
    cudaEventRecord(start);
    kernel_53T<<<blocks,threads>>>(da, db, dc);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("kernel_53T Compute time taken:  %0.3f ms\n", milliseconds);
    seconds = milliseconds / 1000;
    printf("   %0.3f TFlops\n", 2.0*4096*4096*4096/seconds/1e12);

    cublaserr(cublasGetMatrix(4096, 4096, sizeof(float), dc, 4096, c, 4096));
    validate_math(validate, c);
    cublaserr(cublasSetMatrix(4096, 4096, sizeof(float), c, 4096, dc, 4096));
    printf("\n");


    // time kernel_32
    cublaserr(cublasSetMatrix(4096, 4096, sizeof(float), c, 4096, dc, 4096));

    cudaEventRecord(start);
    kernel_32<<<blocks,threads>>>(da, db, dc);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("kernel_32 Compute time taken:  %0.3f ms\n", milliseconds);
    seconds = milliseconds / 1000;
    printf("   %0.3f TFlops\n", 2.0*4096*4096*4096/seconds/1e12);

    cublaserr(cublasGetMatrix(4096, 4096, sizeof(float), dc, 4096, c, 4096));
    validate_math(validate, c);
    cublaserr(cublasSetMatrix(4096, 4096, sizeof(float), c, 4096, dc, 4096));
    printf("\n");




    // time MatMul from nvidia docs
    cublaserr(cublasSetMatrix(4096, 4096, sizeof(float), c, 4096, dc, 4096));

    cudaEventRecord(start);
    MatMul(da, db, dc);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("MatMul Compute time taken:  %0.3f ms\n", milliseconds);
    seconds = milliseconds / 1000;
    printf("   %0.3f TFlops\n", 2.0*4096*4096*4096/seconds/1e12);

    cublaserr(cublasGetMatrix(4096, 4096, sizeof(float), dc, 4096, c, 4096));
    // validate_math(validate, c);
    // nvidia doc code uses row major, so it isn't valid
    // swapping that in their code makes to take 2x as long, so this is just for timing of their crappy code
    cublaserr(cublasSetMatrix(4096, 4096, sizeof(float), c, 4096, dc, 4096));
    printf("\n");


#endif



#if 0
    // time the c version of blas, from the standard netlib repo
    // measuring 4.172 GFlops, currently
    auto hstart = high_resolution_clock::now();
    #define SIZE    2048
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, SIZE, SIZE, SIZE,
        alpha,
        a, SIZE,
        b, SIZE,
        beta,
        c, SIZE);
    auto hstop = high_resolution_clock::now();
    auto duration = duration_cast<std::chrono::milliseconds>(hstop - hstart);
    milliseconds = duration.count();
    printf("blas time taken:  %0.3f\n", milliseconds); 
    seconds = milliseconds / 1000;
    printf("   %0.3f GFlops\n", 2.0*SIZE*SIZE*SIZE/seconds/1e9);
#endif

    cuerr(cudaEventDestroy(start));
    cuerr(cudaEventDestroy(stop));

    cuerr(cudaFree(dvalidate));
    cuerr(cudaFree(da));
    cuerr(cudaFree(db));
    cuerr(cudaFree(dc));


    free(a);
    free(b);
    free(c);
    free(validate);

    cublaserr(cublasDestroy(cublas_handle));
    return 0;
}
