import operator
import math
from typing import Tuple
import numpy as np
from functools import reduce
from . import ndarray_backend_cpu, ndarray_backend_numpy
from .ndarray import prod, cuda, cpu_numpy, cpu, default_device, all_devices


class SparseNDArray:
    """A sparse ndarray class that may contain multiple different backends
    i.e. a numpy backend, a native CPU backend, or a GPU backend.

    We use the Compressed Sparse Row (CSR) format to represent the sparse matrix.
    It is a three-tuple (data, indices, indptr), where:
    - data is an array of the non-zero values in the matrix,
    - indices is an array of the column indices of the non-zero values,
    - indptr is an array of indices into the data and indices arrays
      that point to the start of each row in the matrix.

    For now, the class only supports addition and multiplication of sparse matrices.
    """

    def __init__(self, other, device=None):
        """Create by copying another sparse ndarray, or from numpy ndarray."""
        if isinstance(other, SparseNDArray):
            if device is None:
                device = other.device
            self._init(other.to(device) + 0.0)  # Copy the other sparse ndarray
        elif isinstance(other, np.ndarray):
            # Convert a numpy ndarray to a sparse ndarray
            device = device if device is not None else default_device()
            array = self.make_sparse_from_numpy(other, device)
            self._init(array)
        else:
            raise ValueError(f"Unsupported type {type(other)}")

    def _init(self, other):
        """Initialize SparseNDArray from another SparseNDArray."""
        self._shape = other._shape
        self._device = other._device
        self._csr_array = other._csr_array  # Backend SparseArray object
        self.nnz = self._csr_array.nnz

    @staticmethod
    def make_sparse_from_numpy(ndarray, device):
        """
        Convert a numpy dense array to SparseNDArray in CSR format.
        """
        if ndarray.ndim != 2:
            raise ValueError("Only 2D arrays can be converted to SparseNDArray.")

        rows, cols = ndarray.shape
        indptr = [0]
        indices = []
        data = []

        for i in range(rows):
            for j in range(cols):
                if ndarray[i, j] != 0:
                    indices.append(j)
                    data.append(ndarray[i, j])
            indptr.append(len(indices))

        return SparseNDArray.make((rows, cols), data, indices, indptr, device)

    @staticmethod
    def make(shape, data, indices, indptr, device=None):
        """
        Create a SparseNDArray from raw CSR components.
        """
        array = SparseNDArray.__new__(SparseNDArray)
        array._shape = tuple(shape)
        array._device = device if device is not None else default_device()

        # Create the C++ SparseArray backend
        array._crs_array = device.SparseArray(len(data), shape[0], shape[1])
        array._crs_array.from_components(data, indices, indptr)

        return array

    # def _init(self, other):
    #     self._shape = other._shape
    #     self._device = other._device
    #     self._data = other._data
    #     self._indices = other._indices
    #     self._indptr = other._indptr
    #     self.nnz = len(self._data)

    # @staticmethod
    # def make_sparse_from_numpy(ndarray, device):
    #     """
    #     Convert a numpy dense array to SparseNDArray in CSR format.
    #     """
    #     # Ensure ndarray is 2D
    #     if ndarray.ndim != 2:
    #         raise ValueError("Only 2D arrays can be converted to SparseNDArray.")

    #     rows, cols = ndarray.shape
    #     indptr = [0]
    #     indices = []
    #     data = []

    #     for i in range(rows):
    #         for j in range(cols):
    #             if ndarray[i, j] != 0:
    #                 indices.append(j)
    #                 data.append(ndarray[i, j])
    #         indptr.append(len(indices))

    #     return SparseNDArray.make((rows, cols), data, indices, indptr, device)

    ## The function below is not needed, can be removed after confirmation later
    # @staticmethod
    # def make(shape, data, indices, indptr, device=None):
    #     """Create a sparse ndarray from the given data."""
    #     array = SparseNDArray.__new__(SparseNDArray)
    #     array._shape = tuple(shape)
    #     array._device = device
    #     array._data = data
    #     array._indices = indices  # column index
    #     array._indptr = indptr  # row index
    #     array.nnz = len(data)
    #     return array

    # @staticmethod
    # def make(shape, data, indices, indptr, device=None):
    #     """
    #     Create a SparseNDArray from raw CSR components.
    #     """
    #     array = SparseNDArray.__new__(SparseNDArray)
    #     array._shape = tuple(shape)
    #     array._device = device
    #     # TODO: Use a SparseArray object to store the data
    #     # from NDArray, we can see that only the array of some size is created
    #     # the actual values are stored in the SparseArray object afterwords using the respective operations

    #     # array._crs_array = array.device.SparseArray()

    #     # TODO: we probably dont need this np.ascontiguousarray and we might have to use members of the SparseArray object from C++
    #     array._data = np.ascontiguousarray(data)
    #     array._indices = np.ascontiguousarray(indices)
    #     array._indptr = np.ascontiguousarray(indptr)
    #     array.nnz = len(data)
    #     return array

    @property
    def to_numpy_array(self):
        """Convert the sparse ndarray to a numpy ndarray."""
        array = np.zeros(self._shape, dtype=np.float32)
        num_rows = self._shape[0]
        for i in range(num_rows):
            start = self._indptr[i]
            end = self._indptr[i + 1]
            for j in range(start, end):
                array[i, self._indices[j]] = self._data[j]

        return array

    @staticmethod
    def to_sparse(other: np.ndarray, device=None):
        """Convert a numpy ndarray to a sparse ndarray."""
        shape = other.shape
        assert len(shape) == 2, "only support 2D arrays"

        num_rows = shape[0]
        num_cols = shape[1]

        data = []
        indices = []
        # The size of indptr array is num_rows + 1
        indptr = [0] * (num_rows + 1)

        for i in range(num_rows):
            indptr[i] = len(data)
            for j in range(num_cols):
                if other[i, j] != 0:
                    data.append(other[i, j])
                    indices.append(j)
        indptr[num_rows] = len(data)

        return SparseNDArray.make(shape, data, indices, indptr, device)

    # @staticmethod
    # def to_numpy(other: SparseNDArray):
    #     """Convert a sparse ndarray to a numpy ndarray."""
    #     return other.to_numpy_array()

    @staticmethod
    def create_random_matrix(shape: Tuple[int, int]) -> np.ndarray:
        """Create a random matrix with the given shape."""
        return np.random.randint(low=1, high=100, size=shape)

    ### Properies and string representations
    @property
    def shape(self):
        return self._shape

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        # only support float32 for now
        return "float32"

    @property
    def ndim(self):
        """Return number of dimensions."""
        return len(self._shape)

    @property
    def size(self):
        return prod(self._shape)

    @property
    def count_nonzero(self):
        return self.nnz

    @property
    def to_cpp_sparse_array(self):
        """
        Convert this SparseNDArray to a C++ SparseArray.
        """
        # Retrieve raw pointers and metadata
        data_ptr = self._data.ctypes.data
        indices_ptr = self._indices.ctypes.data
        indptr_ptr = self._indptr.ctypes.data

        # Call into the backend to create a C++ SparseArray
        return SparseArray(
            self.nnz, self._shape[0], self._shape[1], data_ptr, indices_ptr, indptr_ptr
        )

    def __repr__(self):
        return f"SparseNDArray(shape={self._shape}, device={self._device}, dtype={self.dtype})"

    def __str__(self):
        return f"SparseNDArray(shape={self._shape}, device={self._device}, dtype={self.dtype})"

    def to(self, device):
        """Move the sparse ndarray to the given device."""
        if self._device == device:
            return self
        else:
            return SparseNDArray(self.to_numpy_array(), device)

    def broadcast_to(self, shape):
        """Broadcast the sparse ndarray to the given shape."""
        raise NotImplementedError()

    def ewise_or_scalar(self, other, ewise_func, scalar_func):
        """Run either an elementwise or scalar version of a function,
        depending on whether "other" is an SparseNDArray or scalar
        """
        out = SparseNDArray.make(
            self._shape, self._data, self._indices, self._indptr, self._device
        )
        if isinstance(other, SparseNDArray):
            assert self.shape == other.shape, "operation needs two equal-sized arrays"
            # Convert self and other into the expected C++ SparseArray type
            sparse_self = self.to_cpp_sparse_array()
            sparse_other = other.to_cpp_sparse_array()
            sparse_out = out.to_cpp_sparse_array()

            # Call the C++ elementwise function
            ewise_func(sparse_out, sparse_self, sparse_other)

        # Case 2: Scalar operation
        else:
            # Convert self into C++ SparseArray and call scalar_func
            sparse_self = self.to_cpp_sparse_array()
            sparse_out = out.to_cpp_sparse_array()

            scalar_func(sparse_out, sparse_self, float(other))

        return out

    def __add__(self, other):
        return self.ewise_or_scalar(
            other, self.device.sparse_ewise_add, self.device.sparse_scalar_add
        )

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        return self.ewise_or_scalar(
            other, self.device.sparse_ewise_mul, self.device.sparse_scalar_mul
        )

    __rmul__ = __mul__

    def __neg__(self):
        return self * (-1)

    """
    from ndarray_backend, we have these methods:
    
    m.def("sparse_ewise_add", SparseEwiseAdd);
    m.def("sparse_scalar_add", SparseScalarAdd);
    m.def("sparse_ewise_mul", SparseEwiseMul);
    m.def("sparse_scalar_mul", SparseScalarMul);
    m.def("sparse_mat_dense_vec_mul", SparseMatDenseVecMul);
    m.def("sparse_mat_sparse_vec_mul", SparseMatSparseVecMul);
    m.def("sparse_mat_dense_mat_mul", SparseMatDenseMatMul);
    m.def("sparse_mat_sparse_mat_mul", SparseMatSparseMatMul);
    """

    def __matmul__(self, other):
        """Matrix multiplication with another sparse or dense ndarray.
        Like NDArray matmul, we don't handle batch matrix multiplication, and
        require that both arrays be 2D (sizes need to match properly).

        We use the following algorithm:
        1. check if the other is a sparse ndarray

        2. if the other is a sparse ndarray, we use the following algorithm:
            - check if the shapes are compatible
            - create a new sparse ndarray with the correct shape
            - use the device's sparse_matmul method to compute the result

        3. if the other is a dense ndarray, we use the following algorithm:
            - call the device's sparse_matmul_dense method to compute the result
        """

        assert self.ndim == 2 and other.ndim == 2, "matmul requires 2D arrays"
        assert self.shape[1] == other.shape[0], "matmul shape mismatch"

        output_shape = (self.shape[0], other.shape[1])

        if isinstance(other, SparseNDArray):
            # Handle sparse @ sparse case
            if other.shape[1] == 1:
                # Vector case
                return self.device.sparse_mat_sparse_vec_mul(self, other)
            else:
                # Matrix case
                return self.device.sparse_mat_sparse_mat_mul(self, other)
        else:
            # Handle sparse @ dense case
            if other.shape[1] == 1:
                # Vector case
                return self.device.sparse_mat_dense_vec_mul(self, other)
            else:
                # Matrix case
                return self.device.sparse_mat_dense_mat_mul(self, other)


def sparse_add(a, b):
    return a + b


def sparse_mul(a, b):
    return a * b


def add_scalar(a, scalar):
    return a + scalar


def mul_scalar(a, scalar):
    return a * scalar


def broadcast_to(a, shape):
    return a.broadcast_to(shape)
