#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace needle
{
    namespace cpu
    {

#define ALIGNMENT 256
#define TILE 8
        typedef float scalar_t;
        const size_t ELEM_SIZE = sizeof(scalar_t);

        using namespace std;

        /**
         * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
         * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
         * here by default.
         */
        struct AlignedArray
        {
            AlignedArray(const size_t size)
            {
                int ret = posix_memalign((void **)&ptr, ALIGNMENT, size * ELEM_SIZE);
                if (ret != 0)
                    throw std::bad_alloc();
                this->size = size;
            }
            ~AlignedArray() { free(ptr); }
            size_t ptr_as_int() { return (size_t)ptr; }
            scalar_t *ptr;
            size_t size;
        };

        void Fill(AlignedArray *out, scalar_t val)
        {
            /**
             * Fill the values of an aligned array with val
             */
            for (int i = 0; i < out->size; i++)
            {
                out->ptr[i] = val;
            }
        }

        void display_array(const AlignedArray *a)
        {
            for (int i = 0; i < a->size; i++)
            {
                cout << a->ptr[i] << " ";
            }
            cout << endl;
        }

        /*
        enum for get_item and set_item
        */
        enum OpType
        {
            GET,
            SET,
            SCALAR
        };
        void write_array(const AlignedArray *a, AlignedArray *out, const std::vector<int32_t> &shape,
                         int size_a, const std::vector<int32_t> &strides, size_t offset,
                         OpType op_type, scalar_t val = 0)
        {
            std::vector<int32_t> indices(shape.size(), 0); // Multi-dimensional indices
            int cnt = 0;                                   // Counter for elements in the flattened output array

            while (cnt < size_a)
            {
                // Calculate the linear index based on the current multi-dimensional indices and strides
                size_t index = offset;
                for (size_t j = 0; j < strides.size(); j++)
                {
                    index += strides[j] * indices[j];
                }

                // Perform the operation based on op_type
                switch (op_type)
                {
                case GET:
                    out->ptr[cnt] = a->ptr[index];
                    break;
                case SET:
                    out->ptr[index] = a->ptr[cnt];
                    break;
                case SCALAR:
                    out->ptr[index] = val;
                    break;
                }
                cnt++;

                // Update indices for the next element in multi-dimensional space
                for (int i = indices.size() - 1; i >= 0; i--)
                {
                    indices[i]++;
                    if (indices[i] < shape[i])
                    {
                        break; // Move to the next position without resetting lower dimensions
                    }
                    indices[i] = 0; // Reset the current dimension and carry over to the next higher dimension
                }
            }
        }

        // void write_array(const AlignedArray *a, AlignedArray *out, std::vector<int32_t> shape,
        //                  int cnt, int size_a, std::vector<int32_t> indices,
        //                  std::vector<int32_t> strides, size_t offset, OpType op_type, scalar_t val = 0)
        // {
        //     if (cnt >= size_a)
        //     {
        //         return;
        //     }

        //     size_t index = offset;
        //     for (size_t j = 0; j < strides.size(); j++)
        //     {
        //         index += strides[j] * indices[j];
        //     }

        //     switch (op_type)
        //     {
        //     case GET:
        //         out->ptr[cnt] = a->ptr[index];
        //         break;
        //     case SET:
        //         out->ptr[index] = a->ptr[cnt];
        //         break;
        //     case SCALAR:
        //         out->ptr[index] = val;
        //         break;
        //     }
        //     cnt++;

        //     if (size_a == 1)
        //     {
        //         // cout << ">>>> size_a: " << size_a << " cnt: " << cnt << " index: " << index << " offset: " << offset << endl;
        //         return;
        //     }

        //     // Update indices for the next call
        //     for (int i = indices.size() - 1; i >= 0; i--)
        //     {
        //         indices[i]++;
        //         if (indices[i] < shape[i])
        //         {
        //             // Recursive call with incremented cnt and updated indices
        //             write_array(a, out, shape, cnt, size_a, indices, strides, offset, op_type, val);
        //             return;
        //         }
        //         indices[i] = 0;
        //     }
        // }

        void Compact(const AlignedArray &a, AlignedArray *out, std::vector<int32_t> shape,
                     std::vector<int32_t> strides, size_t offset)
        {
            /**
             * Compact an array in memory
             *
             * Args:
             *   a: non-compact representation of the array, given as input
             *   out: compact version of the array to be written
             *   shape: shapes of each dimension for a and out
             *   strides: strides of the *a* array (not out, which has compact strides)
             *   offset: offset of the *a* array (not out, which has zero offset, being compact)
             *
             * Returns:
             *  void (you need to modify out directly, rather than returning anything; this is true for all the
             *  function will implement here, so we won't repeat this note.)
             */
            /// BEGIN SOLUTION
            // assert(false && "Not Implemented");
            vector<int32_t> indices(shape.size(), 0);
            int cnt = 0;
            int size_a = 1;
            for (size_t i = 0; i < shape.size(); i++)
            {
                size_a *= shape[i];
            }

            // write_array(&a, out, shape, cnt, size_a, indices, strides, offset, OpType::GET);
            write_array(&a, out, shape, size_a, strides, offset, OpType::GET);
        }

        void EwiseSetitem(const AlignedArray &a, AlignedArray *out, std::vector<int32_t> shape,
                          std::vector<int32_t> strides, size_t offset)
        {
            /**
             * Set items in a (non-compact) array
             *
             * Args:
             *   a: _compact_ array whose items will be written to out
             *   out: non-compact array whose items are to be written
             *   shape: shapes of each dimension for a and out
             *   strides: strides of the *out* array (not a, which has compact strides)
             *   offset: offset of the *out* array (not a, which has zero offset, being compact)
             */
            /// BEGIN SOLUTION
            // assert(false && "Not Implemented");

            vector<int32_t> indices(shape.size(), 0);
            int cnt = 0;
            int size_a = 1;
            for (size_t i = 0; i < shape.size(); i++)
            {
                size_a *= shape[i];
            }

            // write_array(&a, out, shape, cnt, size_a, indices, strides, offset, OpType::SET);
            write_array(&a, out, shape, size_a, strides, offset, OpType::SET);
            /// END SOLUTION
        }

        void ScalarSetitem(const size_t size, scalar_t val, AlignedArray *out, std::vector<int32_t> shape,
                           std::vector<int32_t> strides, size_t offset)
        {
            /**
             * Set items is a (non-compact) array
             *
             * Args:
             *   size: number of elements to write in out array (note that this will note be the same as
             *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
             *         product of items in shape, but convenient to just pass it here.
             *   val: scalar value to write to
             *   out: non-compact array whose items are to be written
             *   shape: shapes of each dimension of out
             *   strides: strides of the out array
             *   offset: offset of the out array
             */

            /// BEGIN SOLUTION
            // assert(false && "Not Implemented");
            std::vector<int32_t> indices(shape.size(), 0);
            int cnt = 0;
            // write_array(nullptr, out, shape, cnt, size, indices, strides, offset, OpType::SCALAR, val);
            write_array(nullptr, out, shape, size, strides, offset, OpType::SCALAR, val);
            // END SOLUTION
        }

        void EwiseAdd(const AlignedArray &a, const AlignedArray &b, AlignedArray *out)
        {
            /**
             * Set entries in out to be the sum of correspondings entires in a and b.
             */
            for (size_t i = 0; i < a.size; i++)
            {
                out->ptr[i] = a.ptr[i] + b.ptr[i];
            }
        }

        void ScalarAdd(const AlignedArray &a, scalar_t val, AlignedArray *out)
        {
            /**
             * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
             */
            for (size_t i = 0; i < a.size; i++)
            {
                out->ptr[i] = a.ptr[i] + val;
            }
        }

        /**
         * In the code the follows, use the above template to create analogous element-wise
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

        void EwiseMul(const AlignedArray &a, const AlignedArray &b, AlignedArray *out)
        {
            /**
             * Set entries in out to be the product of correspondings entires in a and b.
             */
            for (size_t i = 0; i < a.size; i++)
            {
                out->ptr[i] = a.ptr[i] * b.ptr[i];
            }
        }

        void ScalarMul(const AlignedArray &a, scalar_t val, AlignedArray *out)
        {
            /**
             * Set entries in out to be the product of corresponding entry in a plus the scalar val.
             */
            for (size_t i = 0; i < a.size; i++)
            {
                out->ptr[i] = a.ptr[i] * val;
            }
        }

        void EwiseDiv(const AlignedArray &a, const AlignedArray &b, AlignedArray *out)
        {
            /**
             * Set entries in out to be the division of correspondings entires in a and b.
             */
            for (size_t i = 0; i < a.size; i++)
            {
                out->ptr[i] = a.ptr[i] / b.ptr[i];
            }
        }

        void ScalarDiv(const AlignedArray &a, scalar_t val, AlignedArray *out)
        {
            /**
             * Set entries in out to be the division of corresponding entry in a plus the scalar val.
             */
            for (size_t i = 0; i < a.size; i++)
            {
                out->ptr[i] = a.ptr[i] / val;
            }
        }

        void ScalarPower(const AlignedArray &a, scalar_t val, AlignedArray *out)
        {
            /**
             * Set entries in out to be the power of corresponding entry in a raised to the scalar val.
             */
            for (size_t i = 0; i < a.size; i++)
            {
                out->ptr[i] = std::pow(a.ptr[i], val);
            }
        }

        void EwiseMaximum(const AlignedArray &a, const AlignedArray &b, AlignedArray *out)
        {
            /**
             * Set entries in out to be the maximum of correspondings entires in a and b.
             */
            for (size_t i = 0; i < a.size; i++)
            {
                out->ptr[i] = a.ptr[i] > b.ptr[i] ? a.ptr[i] : b.ptr[i];
            }
        }

        void ScalarMaximum(const AlignedArray &a, scalar_t val, AlignedArray *out)
        {
            /**
             * Set entries in out to be the maximum of corresponding entry in a plus the scalar val.
             */
            for (size_t i = 0; i < a.size; i++)
            {
                out->ptr[i] = a.ptr[i] > val ? a.ptr[i] : val;
            }
        }

        void EwiseEq(const AlignedArray &a, const AlignedArray &b, AlignedArray *out)
        {
            /**
             * Set entries in out to be the equal of correspondings entires in a and b.
             */
            for (size_t i = 0; i < a.size; i++)
            {
                out->ptr[i] = a.ptr[i] == b.ptr[i] ? 1 : 0;
            }
        }

        void ScalarEq(const AlignedArray &a, scalar_t val, AlignedArray *out)
        {
            /**
             * Set entries in out to be the equal of corresponding entry in a plus the scalar val.
             */
            for (size_t i = 0; i < a.size; i++)
            {
                out->ptr[i] = a.ptr[i] == val ? 1 : 0;
            }
        }

        void EwiseGe(const AlignedArray &a, const AlignedArray &b, AlignedArray *out)
        {
            /**
             * Set entries in out to be the greater than equal of correspondings entires in a and b.
             */
            for (size_t i = 0; i < a.size; i++)
            {
                out->ptr[i] = a.ptr[i] >= b.ptr[i] ? 1 : 0;
            }
        }

        void ScalarGe(const AlignedArray &a, scalar_t val, AlignedArray *out)
        {
            /**
             * Set entries in out to be the greater than equal of corresponding entry in a plus the scalar val.
             */
            for (size_t i = 0; i < a.size; i++)
            {
                out->ptr[i] = a.ptr[i] >= val ? 1 : 0;
            }
        }

        void EwiseLog(const AlignedArray &a, AlignedArray *out)
        {
            /**
             * Set entries in out to be the log of correspondings entires in a and b.
             */
            for (size_t i = 0; i < a.size; i++)
            {
                out->ptr[i] = std::log(a.ptr[i]);
            }
        }

        void EwiseExp(const AlignedArray &a, AlignedArray *out)
        {
            /**
             * Set entries in out to be the exp of correspondings entires in a and b.
             */
            for (size_t i = 0; i < a.size; i++)
            {
                out->ptr[i] = std::exp(a.ptr[i]);
            }
        }

        void EwiseTanh(const AlignedArray &a, AlignedArray *out)
        {
            /**
             * Set entries in out to be the tanh of correspondings entires in a and b.
             */
            for (size_t i = 0; i < a.size; i++)
            {
                out->ptr[i] = std::tanh(a.ptr[i]);
            }
        }

        void Matmul(const AlignedArray &a, const AlignedArray &b, AlignedArray *out, uint32_t m, uint32_t n,
                    uint32_t p)
        {
            /**
             * Multiply two (compact) matrices into an output (also compact) matrix.  For this implementation
             * you can use the "naive" three-loop algorithm.
             *
             * Args:
             *   a: compact 2D array of size m x n
             *   b: compact 2D array of size n x p
             *   out: compact 2D array of size m x p to write the output to
             *   m: rows of a / out
             *   n: columns of a / rows of b
             *   p: columns of b / out
             */

            /// BEGIN SOLUTION
            // assert(false && "Not Implemented");

            // initialize out to 0
            for (uint32_t i = 0; i < m; i++)
            {
                for (uint32_t j = 0; j < p; j++)
                {
                    out->ptr[i * p + j] = 0;
                }
            }

            for (uint32_t i = 0; i < m; i++)
            {
                for (uint32_t j = 0; j < p; j++)
                {
                    out->ptr[i * p + j] = 0;
                    for (uint32_t k = 0; k < n; k++)
                    {
                        out->ptr[i * p + j] += a.ptr[i * n + k] * b.ptr[k * p + j];
                    }
                }
            }
            /// END SOLUTION
        }

        inline void AlignedDot(const float *__restrict__ a,
                               const float *__restrict__ b,
                               float *__restrict__ out)
        {

            /**
             * Multiply together two TILE x TILE matrices, and _add _the result to out (it is important to add
             * the result to the existing out, which you should not set to zero beforehand).  We are including
             * the compiler flags here that enable the compile to properly use vector operators to implement
             * this function.  Specifically, the __restrict__ keyword indicates to the compile that a, b, and
             * out don't have any overlapping memory (which is necessary in order for vector operations to be
             * equivalent to their non-vectorized counterparts (imagine what could happen otherwise if a, b,
             * and out had overlapping memory).  Similarly the __builtin_assume_aligned keyword tells the
             * compiler that the input array will be aligned to the appropriate blocks in memory, which also
             * helps the compiler vectorize the code.
             *
             * Args:
             *   a: compact 2D array of size TILE x TILE
             *   b: compact 2D array of size TILE x TILE
             *   out: compact 2D array of size TILE x TILE to write to
             */

            a = (const float *)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
            b = (const float *)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
            out = (float *)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

            /// BEGIN SOLUTION
            // assert(false && "Not Implemented");

            for (uint32_t i = 0; i < TILE; i++)
            {
                for (uint32_t j = 0; j < TILE; j++)
                {
                    float sum = 0;
                    for (uint32_t k = 0; k < TILE; k++)
                    {
                        sum += a[i * TILE + k] * b[k * TILE + j];
                    }
                    out[i * TILE + j] += sum; // Accumulate the sum into out
                }
            }
            /// END SOLUTION
        }

        void MatmulTiled(const AlignedArray &a, const AlignedArray &b, AlignedArray *out, uint32_t m,
                         uint32_t n, uint32_t p)
        {
            /**
             * Matrix multiplication on tiled representations of array.  In this setting, a, b, and out
             * are all *4D* compact arrays of the appropriate size, e.g. a is an array of size
             *   a[m/TILE][n/TILE][TILE][TILE]
             * You should do the multiplication tile-by-tile to improve performance of the array (i.e., this
             * function should call `AlignedDot()` implemented above).
             *
             * Note that this function will only be called when m, n, p are all multiples of TILE, so you can
             * assume that this division happens without any remainder.
             *
             * Args:
             *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE
             *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE
             *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
             *   m: rows of a / out
             *   n: columns of a / rows of b
             *   p: columns of b / out
             *
             */
            /// BEGIN SOLUTION
            // assert(false && "Not Implemented");

            // intialize out to 0
            // Initialize out to 0
            for (uint32_t i = 0; i < (m / TILE) * (p / TILE) * TILE * TILE; i++)
            {
                out->ptr[i] = 0;
            }

            for (uint32_t i = 0; i < m / TILE; i++)
            {
                for (uint32_t j = 0; j < n / TILE; j++)
                {
                    for (uint32_t k = 0; k < p / TILE; k++)
                    {
                        AlignedDot(a.ptr + (i * n / TILE + j) * TILE * TILE,
                                   b.ptr + (j * p / TILE + k) * TILE * TILE,
                                   out->ptr + (i * p / TILE + k) * TILE * TILE);
                    }
                }
            }
        }

        void ReduceMax(const AlignedArray &a, AlignedArray *out, size_t reduce_size)
        {
            /**
             * Reduce by taking maximum over `reduce_size` contiguous blocks.
             *
             * Args:
             *   a: compact array of size a.size = out.size * reduce_size to reduce over
             *   out: compact array to write into
             *   reduce_size: size of the dimension to reduce over
             */

            /// BEGIN SOLUTION
            // assert(false && "Not Implemented");
            for (size_t i = 0; i < out->size; i++)
            {
                out->ptr[i] = a.ptr[i * reduce_size];
                for (size_t j = 1; j < reduce_size; j++)
                {
                    out->ptr[i] = std::max(out->ptr[i], a.ptr[i * reduce_size + j]);
                }
            }

            /// END SOLUTION
        }

        void ReduceSum(const AlignedArray &a, AlignedArray *out, size_t reduce_size)
        {
            /**
             * Reduce by taking sum over `reduce_size` contiguous blocks.
             *
             * Args:
             *   a: compact array of size a.size = out.size * reduce_size to reduce over
             *   out: compact array to write into
             *   reduce_size: size of the dimension to reduce over
             */

            /// BEGIN SOLUTION
            // assert(false && "Not Implemented");

            for (size_t i = 0; i < out->size; i++)
            {
                out->ptr[i] = 0;
                for (size_t j = 0; j < reduce_size; j++)
                {
                    out->ptr[i] += a.ptr[i * reduce_size + j];
                }
            }

            /// END SOLUTION
        }

        /*
        Add spport for sparse matrices.
        Currently we implement addition and multiplcation kernels only.
        For both, if we have two matrices A and B, then at least one of them should be sparse.
        Depending on whether B is sparse or not, we have two different kernels.
        */

        struct SparseArray
        {
            scalar_t *data;  // Non-zero values
            int *indices;    // Column indices
            int *indptr;     // Row pointers
            size_t nnz;      // Number of non-zero elements
            size_t num_rows; // Number of rows
            size_t num_cols; // Number of columns

            SparseArray(size_t nnz, size_t num_rows, size_t num_cols)
                : nnz(nnz), num_rows(num_rows), num_cols(num_cols)
            {
                int ret;
                ret = posix_memalign((void **)&data, ALIGNMENT, nnz * sizeof(scalar_t));
                if (ret != 0)
                    throw std::bad_alloc();
                ret = posix_memalign((void **)&indices, ALIGNMENT, nnz * sizeof(int));
                if (ret != 0)
                    throw std::bad_alloc();
                ret = posix_memalign((void **)&indptr, ALIGNMENT, (num_rows + 1) * sizeof(int));
                if (ret != 0)
                    throw std::bad_alloc();
            }

            ~SparseArray()
            {
                free(data);
                free(indices);
                free(indptr);
            }
        };

        void SparseEwiseAdd(const SparseArray &a, const AlignedArray &b, AlignedArray *out)
        {
            /**
             * Element-wise addition of sparse matrix `a` and dense matrix `b`.
             * Result is stored in dense matrix `out`.
             *
             * Inputs:
             *  - a: Sparse matrix in CSR format.
             *  - b: Dense matrix.
             *  - out: Output dense matrix (must have the same dimensions as `b`).
             */

            // Ensure dimensions match
            assert(out->size == b.size && "Output array size must match dense matrix size");
            assert(out->size == a.num_rows * a.num_cols && "Dimensions of sparse and dense matrices must match");

            // // Initialize the output array to the values of the dense matrix `b`
            // std::memcpy(out->ptr, b.ptr, b.size * sizeof(scalar_t));

            // Perform sparse element-wise addition
            for (size_t i = 0; i < a.num_rows; i++)
            {
                for (int j = a.indptr[i]; j < a.indptr[i + 1]; j++)
                {
                    size_t dense_index = i * a.num_cols + a.indices[j];
                    out->ptr[dense_index] = a.data[j] + b.ptr[dense_index];
                }
            }
        }

        void SparseScalarAdd(const SparseArray &a, scalar_t val, AlignedArray *out)
        {
            /**
             * Scalar addition of sparse matrix `a` with scalar value `val`.
             * Result is stored in dense matrix `out`.
             *
             * Inputs:
             *  - a: Sparse matrix in CSR format.
             *  - val: Scalar value.
             *  - out: Output dense matrix (must have the same dimensions as `a`).
             */

            // Ensure dimensions match
            assert(out->size == a.num_rows * a.num_cols && "Output array size must match sparse matrix size");

            // Perform sparse scalar addition
            for (size_t i = 0; i < a.num_rows; i++)
            {
                for (int j = a.indptr[i]; j < a.indptr[i + 1]; j++)
                {
                    size_t dense_index = i * a.num_cols + a.indices[j];
                    out->ptr[dense_index] = a.data[j] + val;
                }
            }
        }

        void SparseEwiseMul(const SparseArray &a, const AlignedArray &b, AlignedArray *out)
        {
            /**
             * Element-wise multiplication of sparse matrix `a` and dense matrix `b`.
             * Result is stored in dense matrix `out`.
             *
             * Inputs:
             *  - a: Sparse matrix in CSR format.
             *  - b: Dense matrix.
             *  - out: Output dense matrix (must have the same dimensions as `b`).
             */

            // Ensure dimensions match
            assert(out->size == b.size && "Output array size must match dense matrix size");
            assert(out->size == a.num_rows * a.num_cols && "Dimensions of sparse and dense matrices must match");

            // // Initialize the output array to the values of the dense matrix `b`
            // std::memcpy(out->ptr, b.ptr, b.size * sizeof(scalar_t));

            // Perform sparse element-wise multiplication
            for (size_t i = 0; i < a.num_rows; i++)
            {
                for (int j = a.indptr[i]; j < a.indptr[i + 1]; j++)
                {
                    size_t dense_index = i * a.num_cols + a.indices[j];
                    out->ptr[dense_index] = a.data[j] * b.ptr[dense_index];
                }
            }
        }

        void SparseScalarMul(const SparseArray &a, scalar_t val, AlignedArray *out)
        {
            /**
             * Scalar multiplication of sparse matrix `a` with scalar value `val`.
             * Result is stored in dense matrix `out`.
             *
             * Inputs:
             *  - a: Sparse matrix in CSR format.
             *  - val: Scalar value.
             *  - out: Output dense matrix (must have the same dimensions as `a`).
             */

            // Ensure dimensions match
            assert(out->size == a.num_rows * a.num_cols && "Output array size must match sparse matrix size");

            // Perform sparse scalar multiplication
            for (size_t i = 0; i < a.num_rows; i++)
            {
                for (int j = a.indptr[i]; j < a.indptr[i + 1]; j++)
                {
                    size_t dense_index = i * a.num_cols + a.indices[j];
                    out->ptr[dense_index] = a.data[j] * val;
                }
            }
        }

        void SparseMatDenseVecMul(const SparseArray &a, const AlignedArray &b, AlignedArray *out)
        {
            /**
             * Matrix-vector multiplication of sparse matrix `a` and dense vector `b`.
             * Result is stored in dense vector `out`.
             *
             * Inputs:
             *  - a: Sparse matrix in CSR format.
             *  - b: Dense vector.
             *  - out: Output dense vector (must have the same dimensions as `b`).
             */

            // Ensure dimensions match
            assert(out->size == b.size && "Output array size must match dense vector size");
            assert(a.num_cols == b.size && "Number of columns in sparse matrix must match size of dense vector");

            // Perform sparse matrix-vector multiplication
            for (size_t i = 0; i < a.num_rows; i++)
            {
                scalar_t sum = 0;
                for (int j = a.indptr[i]; j < a.indptr[i + 1]; j++)
                {
                    sum += a.data[j] * b.ptr[a.indices[j]];
                }
                out->ptr[i] = sum;
            }
        }

        void SparseMatSparseVecMul(const SparseArray &a, const SparseArray &b, AlignedArray *out)
        {
            /**
             * Matrix-vector multiplication of sparse matrix `a` and sparse vector `b`.
             * Result is stored in dense vector `out`.
             *
             * Inputs:
             *  - a: Sparse matrix in CSR format.
             *  - b: Sparse vector in CSR format.
             *  - out: Output dense vector (must have the same dimensions as `b`).
             */

            // Ensure dimensions match
            assert(out->size == b.num_cols && "Output array size must match sparse vector size");
            assert(a.num_cols == b.num_rows && "Number of columns in sparse matrix must match number of rows in sparse vector");

            // Perform sparse matrix-vector multiplication
            for (size_t i = 0; i < a.num_rows; i++)
            {
                scalar_t sum = 0;
                for (int j = a.indptr[i]; j < a.indptr[i + 1]; j++)
                {
                    sum += a.data[j] * b.data[b.indptr[a.indices[j]]];
                }
                out->ptr[i] = sum;
            }
        }

        // void SparseMatDenseMatMul(const SparseArray &a, const AlignedArray &b, AlignedArray *out)
        // {
        //     /**
        //      * Matrix-matrix multiplication of sparse matrix `a` and dense matrix `b`.
        //      * Result is stored in dense matrix `out`.
        //      *
        //      * Inputs:
        //      *  - a: Sparse matrix in CSR format.
        //      *  - b: Dense matrix.
        //      *  - out: Output dense matrix (must have the same dimensions as `b`).
        //      */

        //     // Ensure dimensions match
        //     assert(out->size == b.size && "Output array size must match dense matrix size");
        //     assert(out->size == a.num_rows * b.num_cols && "Dimensions of sparse and dense matrices must match");

        //     // Perform sparse matrix-matrix multiplication
        //     for (size_t i = 0; i < a.num_rows; i++)
        //     {
        //         for (int j = a.indptr[i]; j < a.indptr[i + 1]; j++)
        //         {
        //             for (size_t k = 0; k < b.num_cols; k++)
        //             {
        //                 out->ptr[i * b.num_cols + k] += a.data[j] * b.ptr[a.indices[j] * b.num_cols + k];
        //             }
        //         }
        //     }
        // }

        void SparseMatDenseMatMul(const SparseArray &a, const AlignedArray &b, AlignedArray *out)
        {
            /**
             * Matrix-matrix multiplication of sparse matrix `a` and dense matrix `b`.
             * Result is stored in dense matrix `out`.
             *
             * Optimized implementation using column-wise vector multiplication.
             *
             * Inputs:
             *  - a: Sparse matrix in CSR format.
             *  - b: Dense matrix.
             *  - out: Output dense matrix.
             */

            // Ensure dimensions match
            assert(out->size == b.size && "Output array size must match dense matrix size");
            assert(out->size == a.num_rows * b.num_cols && "Dimensions of sparse and dense matrices must match");

            // Create temporary arrays for column-wise operations
            AlignedArray col_vec(a.num_cols);
            AlignedArray result_vec(a.num_rows);

            // Process each column of b separately
            for (size_t col = 0; col < b.size / a.num_cols; col++)
            {
                // Extract column from b
                for (size_t i = 0; i < a.num_cols; i++)
                {
                    col_vec.ptr[i] = b.ptr[i * (b.size / a.num_cols) + col];
                }

                // Multiply sparse matrix with this column
                SparseMatDenseVecMul(a, col_vec, &result_vec);

                // Store result in appropriate column of out
                for (size_t i = 0; i < a.num_rows; i++)
                {
                    out->ptr[i * (b.size / a.num_cols) + col] = result_vec.ptr[i];
                }
            }
        }

        void SparseMatSparseMatMul(const SparseArray &a, const SparseArray &b, AlignedArray *out)
        {
            /**
             * Matrix-matrix multiplication of sparse matrix `a` and sparse matrix `b`.
             * Result is stored in dense matrix `out`.
             *
             * Inputs:
             *  - a: Sparse matrix in CSR format.
             *  - b: Sparse matrix in CSR format.
             *  - out: Output dense matrix (must have the same dimensions as `b`).
             */

            // Ensure dimensions match
            assert(out->size == b.num_cols * a.num_rows && "Output array size must match dense matrix size");
            assert(a.num_cols == b.num_rows && "Number of columns in sparse matrix `a` must match number of rows in sparse matrix `b`");

            // Create temporary arrays for column-wise operations
            AlignedArray col_vec(a.num_cols);
            AlignedArray result_vec(a.num_rows);

            // Process each column of b separately
            for (size_t col = 0; col < b.num_cols; col++)
            {
                // Extract column from b
                for (size_t i = 0; i < b.num_rows; i++)
                {
                    col_vec.ptr[i] = b.data[b.indptr[i] + col];
                }

                // Multiply sparse matrix with this column
                SparseMatDenseVecMul(a, col_vec, &result_vec);

                // Store result in appropriate column of out
                for (size_t i = 0; i < a.num_rows; i++)
                {
                    out->ptr[i * b.num_cols + col] = result_vec.ptr[i];
                }
            }
        }

    } // namespace cpu
} // namespace needle

PYBIND11_MODULE(ndarray_backend_cpu, m)
{
    namespace py = pybind11;
    using namespace needle;
    using namespace cpu;

    m.attr("__device_name__") = "cpu";
    m.attr("__tile_size__") = TILE;

    py::class_<AlignedArray>(m, "Array")
        .def(py::init<size_t>(), py::return_value_policy::take_ownership)
        .def("ptr", &AlignedArray::ptr_as_int)
        .def_readonly("size", &AlignedArray::size);

    // return numpy array (with copying for simplicity, otherwise garbage
    // collection is a pain)
    m.def("to_numpy", [](const AlignedArray &a, std::vector<size_t> shape,
                         std::vector<size_t> strides, size_t offset)
          {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset); });

    // convert from numpy (with copying)
    m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray *out)
          { std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE); });

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
    m.def("matmul_tiled", MatmulTiled);

    m.def("reduce_max", ReduceMax);
    m.def("reduce_sum", ReduceSum);

    m.def("sparse_ewise_add", SparseEwiseAdd);
    m.def("sparse_scalar_add", SparseScalarAdd);
    m.def("sparse_ewise_mul", SparseEwiseMul);
    m.def("sparse_scalar_mul", SparseScalarMul);
    m.def("sparse_mat_dense_vec_mul", SparseMatDenseVecMul);
    m.def("sparse_mat_sparse_vec_mul", SparseMatSparseVecMul);
    m.def("sparse_mat_dense_mat_mul", SparseMatDenseMatMul);
    m.def("sparse_mat_sparse_mat_mul", SparseMatSparseMatMul);
}