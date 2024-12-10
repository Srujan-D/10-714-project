#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace py = pybind11;

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
            scalar_t *data = nullptr;
            int *indices = nullptr;
            int *indptr = nullptr;
            size_t nnz = 0;
            size_t num_rows = 0;
            size_t num_cols = 0;

            SparseArray(size_t nnz, size_t num_rows, size_t num_cols)
                : nnz(nnz), num_rows(num_rows), num_cols(num_cols)
            {
                if (posix_memalign((void **)&data, ALIGNMENT, nnz * sizeof(scalar_t)) != 0)
                    throw std::bad_alloc();
                if (posix_memalign((void **)&indices, ALIGNMENT, nnz * sizeof(int)) != 0)
                    throw std::bad_alloc();
                if (posix_memalign((void **)&indptr, ALIGNMENT, (num_rows + 1) * sizeof(int)) != 0)
                    throw std::bad_alloc();

                std::fill(data, data + nnz, 0);
                std::fill(indices, indices + nnz, 0);
                std::fill(indptr, indptr + num_rows + 1, 0);
            }

            ~SparseArray()
            {
                if (data)
                {
                    free(data);
                    data = nullptr;
                }
                if (indices)
                {
                    free(indices);
                    indices = nullptr;
                }
                if (indptr)
                {
                    free(indptr);
                    indptr = nullptr;
                }
            }

            // SparseArray *from_components(py::list data_list, py::list indices_list, py::list indptr_list)
            // {
            //     if (data_list.size() == 0 && indices_list.size() == 0 && indptr_list.size() == 0)
            //     {
            //         // Initialize as empty sparse array
            //         return this;
            //     }

            //     assert(data_list.size() == nnz && "Data size mismatch");
            //     assert(indices_list.size() == nnz && "Indices size mismatch");
            //     assert(indptr_list.size() == num_rows + 1 && "Indptr size mismatch");

            //     for (size_t i = 0; i < nnz; i++)
            //     {
            //         data[i] = py::cast<scalar_t>(data_list[i]);
            //         indices[i] = py::cast<int>(indices_list[i]);
            //     }

            //     for (size_t i = 0; i < num_rows + 1; i++)
            //     {
            //         indptr[i] = py::cast<int>(indptr_list[i]);
            //     }

            //     return this;
            // }
            SparseArray &from_components(py::list data_list, py::list indices_list, py::list indptr_list)
            {
                // Validate input sizes
                if (data_list.size() != nnz)
                {
                    throw std::invalid_argument("Data size mismatch: expected " + std::to_string(nnz) +
                                                ", got " + std::to_string(data_list.size()));
                }
                if (indices_list.size() != nnz)
                {
                    throw std::invalid_argument("Indices size mismatch: expected " + std::to_string(nnz) +
                                                ", got " + std::to_string(indices_list.size()));
                }
                if (indptr_list.size() != num_rows + 1)
                {
                    throw std::invalid_argument("Indptr size mismatch: expected " + std::to_string(num_rows + 1) +
                                                ", got " + std::to_string(indptr_list.size()));
                }

                // Convert Python lists to C++ arrays
                std::vector<scalar_t> data_vector = py::cast<std::vector<scalar_t>>(data_list);
                std::vector<int> indices_vector = py::cast<std::vector<int>>(indices_list);
                std::vector<int> indptr_vector = py::cast<std::vector<int>>(indptr_list);

                // Copy data to internal arrays
                for (size_t i = 0; i < nnz; i++)
                {
                    data[i] = data_vector[i];
                    indices[i] = indices_vector[i];
                }

                for (size_t i = 0; i < num_rows + 1; i++)
                {
                    indptr[i] = indptr_vector[i];
                }

                return *this;
            }

            void from_cpp_components(const vector<scalar_t> &data_list, const vector<int> &indices_list, const vector<int> &indptr_list)
            {
                // fill the sparse array with the given data
                std::fill(this->data, this->data + this->nnz, 0);
                std::fill(this->indices, this->indices + this->nnz, 0);
                std::fill(this->indptr, this->indptr + this->num_rows + 1, 0);

                // Copy the data
                for (size_t i = 0; i < data_list.size(); i++)
                {
                    cout << "data_list[i]: " << data_list[i] << endl;
                    this->data[i] = data_list[i];
                    cout << "data size after copy 1: " << this->data[i] << endl;
                    this->indices[i] = indices_list[i];
                }

                for (size_t i = 0; i < indptr_list.size(); i++)
                {
                    this->indptr[i] = indptr_list[i];
                }

                this->nnz = data_list.size();
                cout << "completed copying" << endl;
            }
        };

        // void from_dense(const scalar_t *dense_matrix, size_t rows, size_t cols)
        // {
        //     /**
        //      * Convert a dense matrix to CSR format and populate SparseArray.
        //      */
        //     size_t nnz_count = 0;

        //     // Calculate nnz and populate CSR arrays
        //     for (size_t i = 0; i < rows; i++)
        //     {
        //         indptr[i] = nnz_count;
        //         for (size_t j = 0; j < cols; j++)
        //         {
        //             scalar_t val = dense_matrix[i * cols + j];
        //             if (val != 0)
        //             {
        //                 data[nnz_count] = val;
        //                 indices[nnz_count] = j;
        //                 nnz_count++;
        //             }
        //         }
        //     }
        //     indptr[rows] = nnz_count;
        // }

        // First overload - Sparse + Dense --> Output is dense
        void SparseEwiseAdd(const SparseArray &a, const AlignedArray &b, AlignedArray *out)
        {
            assert(out->size == b.size && "Output array size must match dense matrix size");
            assert(out->size == a.num_rows * a.num_cols && "Dimensions of sparse and dense matrices must match");

            for (size_t i = 0; i < a.num_rows; i++)
            {
                for (int j = a.indptr[i]; j < a.indptr[i + 1]; j++)
                {
                    size_t dense_index = i * a.num_cols + a.indices[j];
                    out->ptr[dense_index] = a.data[j] + b.ptr[dense_index];
                }
            }
        }

        // Second overload - Dense + Sparse --> Output is dense
        void SparseEwiseAdd(const AlignedArray &a, const SparseArray &b, AlignedArray *out)
        {
            assert(out->size == a.size && "Output array size must match dense matrix size");
            assert(out->size == b.num_rows * b.num_cols && "Dimensions of sparse and dense matrices must match");

            for (size_t i = 0; i < b.num_rows; i++)
            {
                for (int j = b.indptr[i]; j < b.indptr[i + 1]; j++)
                {
                    size_t dense_index = i * b.num_cols + b.indices[j];
                    out->ptr[dense_index] = a.ptr[dense_index] + b.data[j];
                }
            }
        }

        // Third overload - Sparse + Sparse --> Output is dense
        void SparseEWiseAdd(const SparseArray &a, const SparseArray &b, AlignedArray *out)
        {
            assert(out->size == a.num_rows * a.num_cols && "Output array size must match sparse matrix size");
            assert(out->size == b.num_rows * b.num_cols && "Dimensions of sparse matrices must match");

            for (size_t i = 0; i < a.num_rows; i++)
            {
                for (int j = a.indptr[i]; j < a.indptr[i + 1]; j++)
                {
                    size_t dense_index = i * a.num_cols + a.indices[j];
                    out->ptr[dense_index] = a.data[j];
                }
            }

            for (size_t i = 0; i < b.num_rows; i++)
            {
                for (int j = b.indptr[i]; j < b.indptr[i + 1]; j++)
                {
                    size_t dense_index = i * b.num_cols + b.indices[j];
                    out->ptr[dense_index] += b.data[j];
                }
            }
        }

        // Fourth overload - Sparse + Sparse --> Output is sparse
        void SparseEWiseAdd(const SparseArray &a, const SparseArray &b, SparseArray *out)
        {
            assert(out->num_rows == a.num_rows && out->num_cols == a.num_cols && "Output array size must match matrix dimensions");
            assert(a.num_rows == b.num_rows && a.num_cols == b.num_cols && "Input matrices dimensions must match");

            // indptr needs num_rows + 1 elements because:
            // - Elements 0 to num_rows-1 store starting positions for each row
            // - Last element stores total number of non-zero elements
            vector<int> indptr(a.num_rows + 1, 0);
            vector<int> indices;
            vector<scalar_t> data;

            for (size_t i = 0; i < a.num_rows; i++)
            {
                indptr[i] = data.size();

                int j_a = a.indptr[i];
                int j_b = b.indptr[i];

                while (j_a < a.indptr[i + 1] && j_b < b.indptr[i + 1])
                {
                    if (a.indices[j_a] == b.indices[j_b])
                    {
                        data.push_back(a.data[j_a] + b.data[j_b]);
                        cout << "data size after push 1: " << data.size() << endl;
                        indices.push_back(a.indices[j_a]);
                        j_a++;
                        j_b++;
                    }
                    else if (a.indices[j_a] < b.indices[j_b])
                    {
                        data.push_back(a.data[j_a]);
                        cout << "data size after push 2: " << data.size() << endl;
                        indices.push_back(a.indices[j_a]);
                        j_a++;
                    }
                    else
                    {
                        data.push_back(b.data[j_b]);
                        cout << "data size after push 3: " << data.size() << endl;
                        indices.push_back(b.indices[j_b]);
                        j_b++;
                    }
                }

                while (j_a < a.indptr[i + 1])
                {
                    data.push_back(a.data[j_a]);
                    cout << "data size after push 4: " << data.size() << endl;
                    indices.push_back(a.indices[j_a]);
                    j_a++;
                }

                while (j_b < b.indptr[i + 1])
                {
                    data.push_back(b.data[j_b]);
                    cout << "data size after push 5: " << data.size() << endl;
                    indices.push_back(b.indices[j_b]);
                    j_b++;
                }
            }
            indptr[a.num_rows] = data.size();

            out->from_cpp_components(data, indices, indptr);
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

            // Initialize all elements to val
            for (size_t i = 0; i < out->size; i++)
            {
                out->ptr[i] = val;
            }

            // Add the sparse elements
            for (size_t i = 0; i < a.num_rows; i++)
            {
                for (int j = a.indptr[i]; j < a.indptr[i + 1]; j++)
                {
                    size_t dense_index = i * a.num_cols + a.indices[j];
                    out->ptr[dense_index] = a.data[j] + val;
                }
            }
        }

        // First overload - Sparse * Dense --> Output is dense
        void SparseEwiseMul(const SparseArray &a, const AlignedArray &b, AlignedArray *out)
        {
            assert(out->size == b.size && "Output array size must match dense matrix size");
            assert(out->size == a.num_rows * a.num_cols && "Dimensions of sparse and dense matrices must match");

            // Initialize output to zeros since we're only writing non-zero elements
            std::fill(out->ptr, out->ptr + out->size, 0);

            // Multiply only non-zero elements
            for (size_t i = 0; i < a.num_rows; i++)
            {
                for (int j = a.indptr[i]; j < a.indptr[i + 1]; j++)
                {
                    size_t dense_index = i * a.num_cols + a.indices[j];
                    out->ptr[dense_index] = a.data[j] * b.ptr[dense_index];
                }
            }
        }

        // Second overload - Dense * Sparse --> Output is dense
        void SparseEwiseMul(const AlignedArray &a, const SparseArray &b, AlignedArray *out)
        {
            assert(out->size == a.size && "Output array size must match dense matrix size");
            assert(out->size == b.num_rows * b.num_cols && "Dimensions of sparse and dense matrices must match");

            // Initialize output to zeros since we're only writing non-zero elements
            std::fill(out->ptr, out->ptr + out->size, 0);

            // Multiply only non-zero elements
            for (size_t i = 0; i < b.num_rows; i++)
            {
                for (int j = b.indptr[i]; j < b.indptr[i + 1]; j++)
                {
                    size_t dense_index = i * b.num_cols + b.indices[j];
                    out->ptr[dense_index] = a.ptr[dense_index] * b.data[j];
                }
            }
        }

        // Third overload - Sparse * Sparse --> Output is dense
        void SparseEwiseMul(const SparseArray &a, const SparseArray &b, AlignedArray *out)
        {
            assert(out->size == a.num_rows * a.num_cols && "Output array size must match matrix dimensions");
            assert(a.num_rows == b.num_rows && a.num_cols == b.num_cols && "Dimensions of sparse matrices must match");

            // Initialize output to zeros since we're only writing non-zero elements
            std::fill(out->ptr, out->ptr + out->size, 0);

            // For each row
            for (size_t i = 0; i < a.num_rows; i++)
            {
                // For each non-zero element in row i of matrix a
                for (int j_a = a.indptr[i]; j_a < a.indptr[i + 1]; j_a++)
                {
                    int col_a = a.indices[j_a];

                    // For each non-zero element in row i of matrix b
                    for (int j_b = b.indptr[i]; j_b < b.indptr[i + 1]; j_b++)
                    {
                        int col_b = b.indices[j_b];

                        // If both matrices have non-zero elements in the same position
                        if (col_a == col_b)
                        {
                            size_t dense_index = i * a.num_cols + col_a;
                            out->ptr[dense_index] = a.data[j_a] * b.data[j_b];
                        }
                    }
                }
            }
        }

        // Fourth overload - Sparse * Sparse --> Output is sparse
        void SparseEwiseMul(const SparseArray &a, const SparseArray &b, SparseArray *out)
        {
            assert(out->num_rows == a.num_rows && out->num_cols == a.num_cols && "Output array size must match matrix dimensions");

            vector<int> indptr(a.num_rows + 1, 0);
            vector<int> indices;
            vector<scalar_t> data;

            for (size_t i = 0; i < a.num_rows; i++)
            {
                indptr[i] = data.size();

                for (int j_a = a.indptr[i]; j_a < a.indptr[i + 1]; j_a++)
                {
                    int col_a = a.indices[j_a];

                    // For each non-zero element in row i of matrix b
                    for (int j_b = b.indptr[i]; j_b < b.indptr[i + 1]; j_b++)
                    {
                        int col_b = b.indices[j_b];

                        // If both matrices have non-zero elements in the same position
                        if (col_a == col_b)
                        {
                            indices.push_back(col_a);
                            data.push_back(a.data[j_a] * b.data[j_b]);
                            break;
                        }
                    }
                }
            }
            indptr[a.num_rows] = data.size();

            out->from_cpp_components(data, indices, indptr);
        }

        void SparseScalarMul(const SparseArray &a, scalar_t val, SparseArray *out)
        {
            /**
             * Scalar multiplication of sparse matrix `a` with scalar value `val`.
             * Result is stored in sparse matrix `out`.
             *
             * Inputs:
             *  - a: Sparse matrix in CSR format.
             *  - val: Scalar value.
             *  - out: Output sparse matrix (must have the same dimensions as `a`).
             */

            // Ensure dimensions match
            assert(out->num_rows == a.num_rows && out->num_cols == a.num_cols &&
                   "Output dimensions must match input dimensions");

            // Copy over the indices and indptr arrays since structure stays same
            for (size_t i = 0; i < a.nnz; i++)
            {
                out->indices[i] = a.indices[i];
                out->data[i] = a.data[i] * val;
            }

            for (size_t i = 0; i <= a.num_rows; i++)
            {
                out->indptr[i] = a.indptr[i];
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

        void SparseMatSparseVecMul(const SparseArray &a, const SparseArray &b, SparseArray *out)
        {
            /**
             * Matrix-vector multiplication of sparse matrix `a` and sparse vector `b`.
             * Result is stored in sparse vector `out`.
             *
             * Inputs:
             *  - a: Sparse matrix in CSR format.
             *  - b: Sparse vector in CSR format.
             *  - out: Output sparse vector.
             */

            // Ensure dimensions match
            assert(out->num_rows == a.num_rows && "Output rows must match matrix rows");
            assert(a.num_cols == b.num_rows && "Number of columns in matrix must match number of rows in vector");

            vector<int> indptr(a.num_rows + 1, 0);
            vector<int> indices;
            vector<scalar_t> data;

            // Perform sparse matrix-vector multiplication
            for (size_t i = 0; i < a.num_rows; i++)
            {
                scalar_t sum = 0;
                for (int j = a.indptr[i]; j < a.indptr[i + 1]; j++)
                {
                    scalar_t product = a.data[j] * b.data[b.indptr[a.indices[j]]];
                    if (product != 0)
                    {
                        sum += product;
                    }
                }
                if (sum != 0)
                {
                    indices.push_back(i);
                    data.push_back(sum);
                }
                indptr[i + 1] = data.size();
            }

            out->from_cpp_components(data, indices, indptr);
        }

        void SparseMatDenseMatMul(const SparseArray &a, const AlignedArray &b, AlignedArray *out, size_t b_cols)
        {
            /**
             * Matrix-matrix multiplication of sparse matrix `a` and dense matrix `b`.
             * Result is stored in dense matrix `out`.
             *
             * Inputs:
             *  - a: Sparse matrix in CSR format.
             *  - b: Dense matrix (row-major layout).
             *  - out: Output dense matrix.
             *  - b_cols: Number of columns in matrix b.
             */

            // Ensure dimensions match
            assert(out->size == a.num_rows * b_cols && "Output array size must match matrix multiplication dimensions");
            assert(a.num_cols * b_cols == b.size && "Inner dimensions must match for matrix multiplication");

            // Initialize output matrix to zero
            std::fill(out->ptr, out->ptr + out->size, 0);

            // For each row in sparse matrix a
            for (size_t i = 0; i < a.num_rows; i++) {
                // For each non-zero element in row i
                for (int k = a.indptr[i]; k < a.indptr[i + 1]; k++) {
                    scalar_t val_a = a.data[k];
                    int col_a = a.indices[k];

                    // Multiply this element with the corresponding row in dense matrix b
                    for (size_t j = 0; j < b_cols; j++) {
                        out->ptr[i * b_cols + j] += val_a * b.ptr[col_a * b_cols + j];
                    }
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

            // Initialize output matrix to zero
            std::fill(out->ptr, out->ptr + out->size, 0);

            // Iterate through rows of matrix a
            for (size_t i = 0; i < a.num_rows; i++)
            {
                // For each non-zero element in row i of matrix a
                for (int k = a.indptr[i]; k < a.indptr[i + 1]; k++)
                {
                    scalar_t val_a = a.data[k];
                    int col_a = a.indices[k];

                    // For each non-zero element in row col_a of matrix b
                    for (int j = b.indptr[col_a]; j < b.indptr[col_a + 1]; j++)
                    {
                        scalar_t val_b = b.data[j];
                        int col_b = b.indices[j];

                        // Accumulate product in the result matrix
                        out->ptr[i * b.num_cols + col_b] += val_a * val_b;
                    }
                }
            }
        }

        void SparseMatSparseMatMul(const SparseArray &a, const SparseArray &b, SparseArray *out)
        {
            /**
             * Matrix-matrix multiplication of sparse matrix `a` and sparse matrix `b`.
             * Result is stored in sparse matrix `out`.
             */

            // Ensure dimensions match
            assert(out->num_rows == a.num_rows && out->num_cols == b.num_cols && "Output array dimensions must match result size");
            assert(a.num_cols == b.num_rows && "Matrix dimensions do not allow multiplication");

            // Initialize output CSR components
            vector<int> indptr(a.num_rows + 1, 0);
            vector<int> indices;
            vector<scalar_t> data;

            // Temporary storage for row accumulations
            unordered_map<int, scalar_t> row_accumulator;

            for (size_t i = 0; i < a.num_rows; i++)
            {
                row_accumulator.clear(); // Clear accumulator for the current row

                // Iterate through non-zero elements in row `i` of `a`
                for (int j = a.indptr[i]; j < a.indptr[i + 1]; j++)
                {
                    scalar_t val_a = a.data[j];
                    int col_a = a.indices[j];

                    // Multiply with non-zero elements in row `col_a` of `b`
                    for (int k = b.indptr[col_a]; k < b.indptr[col_a + 1]; k++)
                    {
                        int col_b = b.indices[k];
                        scalar_t val_b = b.data[k];

                        // Accumulate the product
                        row_accumulator[col_b] += val_a * val_b;
                    }
                }

                // Add accumulated values to `data` and `indices` for the current row
                vector<pair<int, scalar_t>> sorted_row(row_accumulator.begin(), row_accumulator.end());
                sort(sorted_row.begin(), sorted_row.end()); // Ensure column indices are sorted

                for (const auto &entry : sorted_row)
                {
                    if (abs(entry.second) > 1e-10) // Skip small numerical values for stability
                    {
                        indices.push_back(entry.first);
                        data.push_back(entry.second);
                    }
                }

                // Update `indptr` for the next row
                indptr[i + 1] = data.size();
            }
            cout << "data size: " << data.size() << endl;
            // Assign components to the output sparse array
            out->from_cpp_components(data, indices, indptr);
        }
    };
} // namespace cpu

PYBIND11_MODULE(ndarray_backend_cpu, m)
{

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

    /*
    py::class_<AlignedArray>(m, "Array")
        .def(py::init<size_t>(), py::return_value_policy::take_ownership)
        .def("ptr", &AlignedArray::ptr_as_int)
        .def_readonly("size", &AlignedArray::size);
    */

    // py::class_<SparseArray>(m, "SparseArray")
    //     .def(py::init<size_t, size_t, size_t>(), py::return_value_policy::take_ownership)
    //     .def("from_components", &SparseArray::from_components, py::return_value_policy::reference)
    //     // .def_readonly("data", &SparseArray::data)
    //     // .def_readonly("indices", &SparseArray::indices)
    //     // .def_readonly("indptr", &SparseArray::indptr)
    //     .def_property_readonly("data", [](const SparseArray &self)
    //                            { return py::cast(std::vector<scalar_t>(self.data, self.data + self.nnz)); })
    //     .def_property_readonly("indices", [](const SparseArray &self)
    //                            { return py::cast(std::vector<int>(self.indices, self.indices + self.nnz)); })
    //     .def_property_readonly("indptr", [](const SparseArray &self)
    //                            { return py::cast(std::vector<int>(self.indptr, self.indptr + self.num_rows + 1)); })

    //     .def_readonly("nnz", &SparseArray::nnz)
    //     .def_readonly("num_rows", &SparseArray::num_rows)
    //     .def_readonly("num_cols", &SparseArray::num_cols);

    py::class_<SparseArray>(m, "SparseArray")
        .def(py::init<size_t, size_t, size_t>(), py::return_value_policy::take_ownership)
        .def("from_components", &SparseArray::from_components, py::return_value_policy::reference,
             "Initialize the sparse array from components.",
             py::arg("data_list"), py::arg("indices_list"), py::arg("indptr_list"))
        .def_property_readonly("data", [](const SparseArray &self)
                               { return py::array_t<scalar_t>(
                                     {self.nnz},         // Shape
                                     {sizeof(scalar_t)}, // Stride
                                     self.data,          // Pointer to data
                                     py::cast(self)      // Ensure SparseArray stays alive
                                 ); })
        .def_property_readonly("indices", [](const SparseArray &self)
                               { return py::array_t<int>(
                                     {self.nnz},    // Shape
                                     {sizeof(int)}, // Stride
                                     self.indices,  // Pointer to indices
                                     py::cast(self) // Ensure SparseArray stays alive
                                 ); })
        .def_property_readonly("indptr", [](const SparseArray &self)
                               { return py::array_t<int>(
                                     {self.num_rows + 1}, // Shape
                                     {sizeof(int)},       // Stride
                                     self.indptr,         // Pointer to indptr
                                     py::cast(self)       // Ensure SparseArray stays alive
                                 ); })
        .def_readonly("nnz", &SparseArray::nnz, "Number of non-zero elements")
        .def_readonly("num_rows", &SparseArray::num_rows, "Number of rows in the sparse matrix")
        .def_readonly("num_cols", &SparseArray::num_cols, "Number of columns in the sparse matrix");

    m.def("sparse_ewise_add_SDD", static_cast<void (*)(const SparseArray &, const AlignedArray &, AlignedArray *)>(&SparseEwiseAdd));
    m.def("sparse_ewise_add_DSD", static_cast<void (*)(const AlignedArray &, const SparseArray &, AlignedArray *)>(&SparseEwiseAdd));
    m.def("sparse_ewise_add_SSD", static_cast<void (*)(const SparseArray &, const SparseArray &, AlignedArray *)>(&SparseEWiseAdd));
    m.def("sparse_ewise_add_SSS", static_cast<void (*)(const SparseArray &, const SparseArray &, SparseArray *)>(&SparseEWiseAdd));

    m.def("sparse_scalar_add", SparseScalarAdd);

    m.def("sparse_ewise_mul_SDD", static_cast<void (*)(const SparseArray &, const AlignedArray &, AlignedArray *)>(&SparseEwiseMul));
    m.def("sparse_ewise_mul_DSD", static_cast<void (*)(const AlignedArray &, const SparseArray &, AlignedArray *)>(&SparseEwiseMul));
    m.def("sparse_ewise_mul_SSD", static_cast<void (*)(const SparseArray &, const SparseArray &, AlignedArray *)>(&SparseEwiseMul));
    m.def("sparse_ewise_mul_SSS", static_cast<void (*)(const SparseArray &, const SparseArray &, SparseArray *)>(&SparseEwiseMul));

    m.def("sparse_scalar_mul", SparseScalarMul);
    m.def("sparse_mat_dense_vec_mul", SparseMatDenseVecMul);
    m.def("sparse_mat_sparse_vec_mul", SparseMatSparseVecMul);
    m.def("sparse_mat_dense_mat_mul", SparseMatDenseMatMul);
    m.def("sparse_mat_sparse_mat_mul", static_cast<void (*)(const SparseArray &, const SparseArray &, AlignedArray *)>(&SparseMatSparseMatMul));
    m.def("sparse_mat_sparse_mat_mul_sparse", static_cast<void (*)(const SparseArray &, const SparseArray &, SparseArray *)>(&SparseMatSparseMatMul));
}