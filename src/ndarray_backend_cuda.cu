#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle
{
    namespace cuda
    {

#define BASE_THREAD_NUM 256

#define TILE 4
        typedef float scalar_t;
        const size_t ELEM_SIZE = sizeof(scalar_t);

        struct CudaArray
        {
            CudaArray(const size_t size)
            {
                cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
                if (err != cudaSuccess)
                    throw std::runtime_error(cudaGetErrorString(err));
                this->size = size;
            }
            ~CudaArray() { cudaFree(ptr); }
            size_t ptr_as_int() { return (size_t)ptr; }

            scalar_t *ptr;
            size_t size;
        };

        struct CudaDims
        {
            dim3 block, grid;
        };

        CudaDims CudaOneDim(size_t size)
        {
            /**
             * Utility function to get cuda dimensions for 1D call
             */
            CudaDims dim;
            size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
            dim.block = dim3(BASE_THREAD_NUM, 1, 1);
            dim.grid = dim3(num_blocks, 1, 1);
            return dim;
        }

        // CudaDums CudaTwoDim()

#define MAX_VEC_SIZE 8
        struct CudaVec
        {
            uint32_t size;
            int32_t data[MAX_VEC_SIZE];
        };

        CudaVec VecToCuda(const std::vector<int32_t> &x)
        {
            CudaVec shape;
            if (x.size() > MAX_VEC_SIZE)
                throw std::runtime_error("Exceeded CUDA supported max dimesions");
            shape.size = x.size();
            for (size_t i = 0; i < x.size(); i++)
            {
                shape.data[i] = x[i];
            }
            return shape;
        }

        ////////////////////////////////////////////////////////////////////////////////
        // Fill call
        ////////////////////////////////////////////////////////////////////////////////

        __global__ void FillKernel(scalar_t *out, scalar_t val, size_t size)
        {
            size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
            if (gid < size)
                out[gid] = val;
        }

        void Fill(CudaArray *out, scalar_t val)
        {
            CudaDims dim = CudaOneDim(out->size);
            FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
        }

        ////////////////////////////////////////////////////////////////////////////////
        // Compact and setitem cals
        ////////////////////////////////////////////////////////////////////////////////

        // Utility function to convert contiguous index i to memory location from strides

        enum OpType
        {
            GET,
            SET,
            SCALAR
        };

        __device__ void write_array(const scalar_t *a, scalar_t *out, CudaVec shape,
                                    int cnt, int size_a, CudaVec indices,
                                    CudaVec strides, size_t offset, OpType op_type, scalar_t val = 0)
        {
            if (cnt >= size_a)
                return;

            // Calculate the flat index using offset and strides
            size_t index = offset;
            for (size_t j = 0; j < strides.size; j++)
                index += strides.data[j] * indices.data[j];

            // Apply the operation based on OpType
            switch (op_type)
            {
            case GET:
                out[cnt] = a[index];
                break;
            case SET:
                out[index] = a[cnt];
                break;
            case SCALAR:
                out[index] = val;
                break;
            }
        }

        __global__ void CompactKernel(const scalar_t *a, scalar_t *out, size_t size, CudaVec shape,
                                      CudaVec strides, size_t offset)
        {
            size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

            if (gid < size)
            {
                CudaVec indices;
                indices.size = shape.size;
                size_t remaining = gid;

                for (int i = shape.size - 1; i >= 0; i--)
                {
                    indices.data[i] = remaining % shape.data[i];
                    remaining /= shape.data[i];
                }

                write_array(a, out, shape, gid, size, indices, strides, offset, GET);
            }
        }

        void Compact(const CudaArray &a, CudaArray *out, std::vector<int32_t> shape,
                     std::vector<int32_t> strides, size_t offset)
        {
            CudaDims dim = CudaOneDim(out->size);
            CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                                   VecToCuda(strides), offset);
        }

        __global__ void EwiseSetitemKernel(const scalar_t *a, scalar_t *out, size_t size, CudaVec shape,
                                           CudaVec strides, size_t offset)

        {
            size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

            if (gid < size)
            {
                CudaVec indices;
                indices.size = shape.size;
                size_t remaining = gid;

                for (int i = shape.size - 1; i >= 0; i--)
                {
                    indices.data[i] = remaining % shape.data[i];
                    remaining /= shape.data[i];
                }
                write_array(a, out, shape, gid, size, indices, strides, offset, SET);
            }
        }

        void EwiseSetitem(const CudaArray &a, CudaArray *out, std::vector<int32_t> shape,
                          std::vector<int32_t> strides, size_t offset)
        {
            /**
             * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
             * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
             *
             * Args:
             *   a: _compact_ array whose items will be written to out
             *   out: non-compact array whose items are to be written
             *   shape: shapes of each dimension for a and out
             *   strides: strides of the *out* array (not a, which has compact strides)
             *   offset: offset of the *out* array (not a, which has zero offset, being compact)
             */
            /// BEGIN SOLUTION
            // assert(false && "Not Implemented")
            CudaDims dim = CudaOneDim(out->size);
            EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape),
                                                        VecToCuda(strides), offset);
            /// END SOLUTION
        }

        __global__ void ScalarSetitemKernel(scalar_t *out, size_t size, CudaVec shape,
                                            CudaVec strides, size_t offset, scalar_t val)
        {
            size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
            if (gid < size)
            {
                CudaVec indices;
                indices.size = shape.size;
                size_t remaining = gid;

                for (int i = shape.size - 1; i >= 0; i--)
                {
                    indices.data[i] = remaining % shape.data[i];
                    remaining /= shape.data[i];
                }

                write_array(nullptr, out, shape, gid, size, indices, strides, offset, SCALAR, val);
            }
        }

        void ScalarSetitem(size_t size, scalar_t val, CudaArray *out, std::vector<int32_t> shape,
                           std::vector<int32_t> strides, size_t offset)
        {
            /**
             * Set items is a (non-compact) array
             *
             * Args:
             *   size: number of elements to write in out array (note that this will note be the same as
             *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
             *         product of items in shape, but covenient to just pass it here.
             *   val: scalar value to write to
             *   out: non-compact array whose items are to be written
             *   shape: shapes of each dimension of out
             *   strides: strides of the out array
             *   offset: offset of the out array
             */
            /// BEGIN SOLUTION
            // assert(false && "Not Implemented");
            CudaDims dim = CudaOneDim(size);
            ScalarSetitemKernel<<<dim.grid, dim.block>>>(out->ptr, size, VecToCuda(shape),
                                                         VecToCuda(strides), offset, val);
            /// END SOLUTION
        }

        ////////////////////////////////////////////////////////////////////////////////
        // Elementwise and scalar operations
        ////////////////////////////////////////////////////////////////////////////////

        __global__ void EwiseAddKernel(const scalar_t *a, const scalar_t *b, scalar_t *out, size_t size)
        {
            // Calculate the global index of the thread.
            size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
            if (gid < size)
                out[gid] = a[gid] + b[gid];
        }

        void EwiseAdd(const CudaArray &a, const CudaArray &b, CudaArray *out)
        {
            /**
             * Add together two CUDA arrays.
             * Args:
             *   a: Input array 'a' to be added
             *   b: Input array 'b' to be added
             *   out: Output array to store the result of 'a + b'
             */
            CudaDims dim = CudaOneDim(out->size);

            // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
            EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
        }

        __global__ void ScalarAddKernel(const scalar_t *a, scalar_t val, scalar_t *out, size_t size)
        {
            // Calculate the global index of the thread.
            size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
            if (gid < size)
                out[gid] = a[gid] + val;
        }

        void ScalarAdd(const CudaArray &a, scalar_t val, CudaArray *out)
        {
            /**
             * Add a scalar value to every element of a CUDA array.
             * Args:
             *   a: Input array 'a'
             *   val: Scalar value to be added
             *   out: Output array to store the result of 'a + val'
             */
            CudaDims dim = CudaOneDim(out->size);

            // Launch the ScalarAddKernel that will add the scalar 'val' to each element of array 'a',
            // and store the result in array 'out'.
            ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
        }

        /**
         * In the code the follows, use the above template to create analogous elementise
         * and and scalar operators for the following functions.  See the numpy backend for
         * examples of how they should work.
         *   - EwiseMul, ScalarMul
         *   - EwiseDiv, ScalarDiv
         *   - ScalarPower
         *   - EwiseMaximum, ScalarMaximum
         *   - EwiseEq, ScalarEq
         *   - EwiseGe, ScalarGe
         *   - EwiseLog
         *   - EwiseExp
         *   - EwiseTanh
         *
         * If you implement all these naively, there will be a lot of repeated code, so
         * you are welcome (but not required), to use macros or templates to define these
         * functions (however you want to do so, as long as the functions match the proper)
         * signatures above.
         */

        ////////////////////////////////////////////////////////////////////////////////
        // Elementwise and scalar operations
        ////////////////////////////////////////////////////////////////////////////////

        __global__ void EwiseMulKernel(const scalar_t *a, const scalar_t *b, scalar_t *out, size_t size)
        {
            // Calculate the global index of the thread.
            size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
            if (gid < size)
                out[gid] = a[gid] * b[gid];
        }

        void EwiseMul(const CudaArray &a, const CudaArray &b, CudaArray *out)
        {
            /**
             * Mul together two CUDA arrays.
             * Args:
             *   a: Input array 'a' to be added
             *   b: Input array 'b' to be added
             *   out: Output array to store the result of 'a + b'
             */
            CudaDims dim = CudaOneDim(out->size);

            // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
            EwiseMulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
        }

        __global__ void ScalarMulKernel(const scalar_t *a, scalar_t val, scalar_t *out, size_t size)
        {
            // Calculate the global index of the thread.
            size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
            if (gid < size)
                out[gid] = a[gid] * val;
        }

        void ScalarMul(const CudaArray &a, scalar_t val, CudaArray *out)
        {
            /**
             * Mul a scalar value to every element of a CUDA array.
             * Args:
             *   a: Input array 'a'
             *   val: Scalar value to be added
             *   out: Output array to store the result of 'a + val'
             */
            CudaDims dim = CudaOneDim(out->size);

            // Launch the ScalarMulKernel that will add the scalar 'val' to each element of array 'a',
            // and store the result in array 'out'.
            ScalarMulKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
        }

        __global__ void EwiseDivKernel(const scalar_t *a, const scalar_t *b, scalar_t *out, size_t size)
        {
            // Calculate the global index of the thread.
            size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
            if (gid < size)
                out[gid] = a[gid] / b[gid];
        }

        void EwiseDiv(const CudaArray &a, const CudaArray &b, CudaArray *out)
        {
            /**
             * Div together two CUDA arrays.
             * Args:
             *   a: Input array 'a' to be added
             *   b: Input array 'b' to be added
             *   out: Output array to store the result of 'a + b'
             */
            CudaDims dim = CudaOneDim(out->size);

            // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
            EwiseDivKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
        }

        __global__ void ScalarDivKernel(const scalar_t *a, scalar_t val, scalar_t *out, size_t size)
        {
            // Calculate the global index of the thread.
            size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
            if (gid < size)
                out[gid] = a[gid] / val;
        }

        void ScalarDiv(const CudaArray &a, scalar_t val, CudaArray *out)
        {
            /**
             * Div a scalar value to every element of a CUDA array.
             * Args:
             *   a: Input array 'a'
             *   val: Scalar value to be added
             *   out: Output array to store the result of 'a + val'
             */
            CudaDims dim = CudaOneDim(out->size);

            // Launch the ScalarDivKernel that will add the scalar 'val' to each element of array 'a',
            // and store the result in array 'out'.
            ScalarDivKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
        }

        __global__ void ScalarPowerKernel(const scalar_t *a, scalar_t val, scalar_t *out, size_t size)
        {
            // Calculate the global index of the thread.
            size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
            if (gid < size)
                out[gid] = pow(a[gid], val);
        }

        void ScalarPower(const CudaArray &a, scalar_t val, CudaArray *out)
        {
            /**
             * Power a scalar value to every element of a CUDA array.
             * Args:
             *   a: Input array 'a'
             *   val: Scalar value to be added
             *   out: Output array to store the result of 'a + val'
             */
            CudaDims dim = CudaOneDim(out->size);

            // Launch the ScalarPowerKernel that will add the scalar 'val' to each element of array 'a',
            // and store the result in array 'out'.
            ScalarPowerKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
        }

        __global__ void EwiseMaximumKernel(const scalar_t *a, const scalar_t *b, scalar_t *out, size_t size)
        {
            // Calculate the global index of the thread.
            size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
            if (gid < size)
                out[gid] = a[gid] > b[gid] ? a[gid] : b[gid];
        }

        void EwiseMaximum(const CudaArray &a, const CudaArray &b, CudaArray *out)
        {
            /**
             * Maximum together two CUDA arrays.
             * Args:
             *   a: Input array 'a' to be added
             *   b: Input array 'b' to be added
             *   out: Output array to store the result of 'a + b'
             */
            CudaDims dim = CudaOneDim(out->size);

            // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
            EwiseMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
        }

        __global__ void ScalarMaximumKernel(const scalar_t *a, scalar_t val, scalar_t *out, size_t size)
        {
            // Calculate the global index of the thread.
            size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
            if (gid < size)
                out[gid] = a[gid] > val ? a[gid] : val;
        }

        void ScalarMaximum(const CudaArray &a, scalar_t val, CudaArray *out)
        {
            /**
             * Maximum a scalar value to every element of a CUDA array.
             * Args:
             *   a: Input array 'a'
             *   val: Scalar value to be added
             *   out: Output array to store the result of 'a + val'
             */
            CudaDims dim = CudaOneDim(out->size);

            // Launch the ScalarMaximumKernel that will add the scalar 'val' to each element of array 'a',
            // and store the result in array 'out'.
            ScalarMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
        }

        __global__ void EwiseEqKernel(const scalar_t *a, const scalar_t *b, scalar_t *out, size_t size)
        {
            // Calculate the global index of the thread.
            size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
            if (gid < size)
                out[gid] = a[gid] == b[gid] ? 1 : 0;
        }

        void EwiseEq(const CudaArray &a, const CudaArray &b, CudaArray *out)
        {
            /**
             * Eq together two CUDA arrays.
             * Args:
             *   a: Input array 'a' to be added
             *   b: Input array 'b' to be added
             *   out: Output array to store the result of 'a + b'
             */
            CudaDims dim = CudaOneDim(out->size);

            // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
            EwiseEqKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
        }

        __global__ void ScalarEqKernel(const scalar_t *a, scalar_t val, scalar_t *out, size_t size)
        {
            // Calculate the global index of the thread.
            size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
            if (gid < size)
                out[gid] = a[gid] == val ? 1 : 0;
        }

        void ScalarEq(const CudaArray &a, scalar_t val, CudaArray *out)
        {
            /**
             * Eq a scalar value to every element of a CUDA array.
             * Args:
             *   a: Input array 'a'
             *   val: Scalar value to be added
             *   out: Output array to store the result of 'a + val'
             */
            CudaDims dim = CudaOneDim(out->size);

            // Launch the ScalarEqKernel that will add the scalar 'val' to each element of array 'a',
            // and store the result in array 'out'.
            ScalarEqKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
        }

        __global__ void EwiseGeKernel(const scalar_t *a, const scalar_t *b, scalar_t *out, size_t size)
        {
            // Calculate the global index of the thread.
            size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
            if (gid < size)
                out[gid] = a[gid] >= b[gid] ? 1 : 0;
        }

        void EwiseGe(const CudaArray &a, const CudaArray &b, CudaArray *out)
        {
            /**
             * Ge together two CUDA arrays.
             * Args:
             *   a: Input array 'a' to be added
             *   b: Input array 'b' to be added
             *   out: Output array to store the result of 'a + b'
             */
            CudaDims dim = CudaOneDim(out->size);

            // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
            EwiseGeKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
        }

        __global__ void ScalarGeKernel(const scalar_t *a, scalar_t val, scalar_t *out, size_t size)
        {
            // Calculate the global index of the thread.
            size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
            if (gid < size)
                out[gid] = a[gid] >= val ? 1 : 0;
        }

        void ScalarGe(const CudaArray &a, scalar_t val, CudaArray *out)
        {
            /**
             * Ge a scalar value to every element of a CUDA array.
             * Args:
             *   a: Input array 'a'
             *   val: Scalar value to be added
             *   out: Output array to store the result of 'a + val'
             */
            CudaDims dim = CudaOneDim(out->size);

            // Launch the ScalarGeKernel that will add the scalar 'val' to each element of array 'a',
            // and store the result in array 'out'.
            ScalarGeKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
        }

        __global__ void EwiseLogKernel(const scalar_t *a, scalar_t *out, size_t size)
        {
            // Calculate the global index of the thread.
            size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
            if (gid < size)
                out[gid] = log(a[gid]);
        }

        void EwiseLog(const CudaArray &a, CudaArray *out)
        {
            /**
             * Log of CUDA arrays.
             * Args:
             *   a: Input array 'a' to be added
             *   out: Output array to store the result of 'log(a)'
             */
            CudaDims dim = CudaOneDim(out->size);

            // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
            EwiseLogKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
        }

        __global__ void EwiseExpKernel(const scalar_t *a, scalar_t *out, size_t size)
        {
            // Calculate the global index of the thread.
            size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
            if (gid < size)
                out[gid] = exp(a[gid]);
        }

        void EwiseExp(const CudaArray &a, CudaArray *out)
        {
            /**
             * Exp of CUDA arrays.
             * Args:
             *   a: Input array 'a' to be added
             *   out: Output array to store the result of 'exp(a)'
             */
            CudaDims dim = CudaOneDim(out->size);

            // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
            EwiseExpKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
        }

        __global__ void EwiseTanhKernel(const scalar_t *a, scalar_t *out, size_t size)
        {
            // Calculate the global index of the thread.
            size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
            if (gid < size)
                out[gid] = tanh(a[gid]);
        }

        void EwiseTanh(const CudaArray &a, CudaArray *out)
        {
            /**
             * Tanh of CUDA arrays.
             * Args:
             *   a: Input array 'a' to be added
             *   out: Output array to store the result of 'tanh(a)'
             */
            CudaDims dim = CudaOneDim(out->size);

            // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
            EwiseTanhKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
        }

        __global__ void MatmulKernel(const scalar_t *A, const scalar_t *B, scalar_t *C, uint32_t M, uint32_t N, uint32_t P)
        {
            int yblock = blockIdx.y;
            int xblock = blockIdx.x;

            int local_y = threadIdx.y;
            int local_x = threadIdx.x;

            __shared__ scalar_t sA[TILE][TILE];
            __shared__ scalar_t sB[TILE][TILE];

            scalar_t c_val = 0;

            for (int ko = 0; ko < (N + TILE - 1) / TILE; ko++)
            {
                int row = yblock * TILE + local_y;
                int col = ko * TILE + local_x;
                if (row < M && col < N)
                {
                    sA[local_y][local_x] = A[row * N + col];
                }
                else
                {
                    sA[local_y][local_x] = 0.0f;
                }

                row = ko * TILE + local_y;
                col = xblock * TILE + local_x;
                if (row < N && col < P)
                {
                    sB[local_y][local_x] = B[row * P + col];
                }
                else
                {
                    sB[local_y][local_x] = 0.0f;
                }

                __syncthreads();

                for (int n = 0; n < TILE; n++)
                {
                    c_val += sA[local_y][n] * sB[n][local_x];
                }

                __syncthreads();
            }

            int out_row = yblock * TILE + local_y;
            int out_col = xblock * TILE + local_x;
            if (out_row < M && out_col < P)
            {
                C[out_row * P + out_col] = c_val;
            }
        }

        void Matmul(const CudaArray &a, const CudaArray &b, CudaArray *out, uint32_t M, uint32_t N, uint32_t P)
        {
            dim3 blockDim(TILE, TILE);
            dim3 gridDim((P + TILE - 1) / TILE, (M + TILE - 1) / TILE);
            MatmulKernel<<<gridDim, blockDim>>>(a.ptr, b.ptr, out->ptr, M, N, P);
        }

        ////////////////////////////////////////////////////////////////////////////////
        // Max and sum reductions
        ////////////////////////////////////////////////////////////////////////////////

        __global__ void ReduceMaxKernel(const scalar_t *a, scalar_t *out, size_t out_size, size_t reduce_size)
        {
            size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
            if (gid < out_size) // Ensure `gid` is within the bounds of `out`
            {
                size_t idx = gid * reduce_size;
                scalar_t max_val = a[idx];
                for (size_t i = 1; i < reduce_size; i++)
                    max_val = max(max_val, a[idx + i]);
                out[gid] = max_val;
            }
        }

        void ReduceMax(const CudaArray &a, CudaArray *out, size_t reduce_size)
        {
            // Ensure we only launch enough threads for `out.size`
            CudaDims dim = CudaOneDim(out->size);
            ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
        }

        __global__ void ReduceSumKernel(const scalar_t *a, scalar_t *out, size_t out_size, size_t reduce_size)
        {
            size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
            if (gid < out_size) // Ensure `gid` is within the bounds of `out`
            {
                scalar_t sum_val = 0;
                size_t idx = gid * reduce_size;
                for (size_t i = 0; i < reduce_size; i++)
                    sum_val += a[idx + i];
                out[gid] = sum_val;
            }
        }

        void ReduceSum(const CudaArray &a, CudaArray *out, size_t reduce_size)
        {
            // Ensure we only launch enough threads for `out.size`
            CudaDims dim = CudaOneDim(out->size);
            ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
        }

    } // namespace cuda
} // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m)
{
    namespace py = pybind11;
    using namespace needle;
    using namespace cuda;

    m.attr("__device_name__") = "cuda";
    m.attr("__tile_size__") = TILE;

    py::class_<CudaArray>(m, "Array")
        .def(py::init<size_t>(), py::return_value_policy::take_ownership)
        .def_readonly("size", &CudaArray::size)
        .def("ptr", &CudaArray::ptr_as_int);

    // return numpy array, copying from CPU
    m.def("to_numpy", [](const CudaArray &a, std::vector<size_t> shape, std::vector<size_t> strides,
                         size_t offset)
          {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer); });

    // copy numpy array to GPU
    m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray *out)
          {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err)); });

    m.def("fill", Fill);
    m.def("compact", Compact);
    m.def("ewise_setitem", EwiseSetitem);
    m.def("scalar_setitem", ScalarSetitem);
    m.def("ewise_add", EwiseAdd);
    m.def("scalar_add", ScalarAdd);

    m.def("ewise_mul", EwiseMul);
    m.def("scalar_mul", ScalarMul);
    m.def("ewise_div", EwiseDiv);
    m.def("scalar_div", ScalarDiv);
    m.def("scalar_power", ScalarPower);

    m.def("ewise_maximum", EwiseMaximum);
    m.def("scalar_maximum", ScalarMaximum);
    m.def("ewise_eq", EwiseEq);
    m.def("scalar_eq", ScalarEq);
    m.def("ewise_ge", EwiseGe);
    m.def("scalar_ge", ScalarGe);

    m.def("ewise_log", EwiseLog);
    m.def("ewise_exp", EwiseExp);
    m.def("ewise_tanh", EwiseTanh);

    m.def("matmul", Matmul);

    m.def("reduce_max", ReduceMax);
    m.def("reduce_sum", ReduceSum);
}