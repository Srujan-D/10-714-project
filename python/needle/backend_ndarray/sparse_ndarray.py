import operator
import math
from typing import Tuple
import numpy as np
from functools import reduce
from . import ndarray_backend_cpu, ndarray_backend_numpy
from .ndarray import (
    prod,
    cuda,
    cpu_numpy,
    cpu,
    default_device,
    all_devices,
    NDArray,
    full,
)


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
        elif isinstance(other, tuple):
            device = device if device is not None else default_device()
            self._shape = other  # shape of the sparse array
            self._device = device
            self._data = []
            self._csr_array = self._device.SparseArray(
                0, self._shape[0], self._shape[1]
            )
            self.nnz = 0

            # raise ValueError(f"Unsupported type {type(other)}")

    def _init(self, other):
        """Initialize SparseNDArray from another SparseNDArray."""
        self._shape = other._shape
        self._device = other._device
        self._csr_array = other._csr_array  # Backend SparseArray object
        self.nnz = self._csr_array.nnz
        self._data = other._data

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

        array._data = data

        # Create the C++ SparseArray backend
        # breakpoint()
        array._csr_array = device.SparseArray(len(data), shape[0], shape[1])
        if device.name == 'cpu':
            # print("Using ", device.name)
            if isinstance(data, np.ndarray):
                data = data.tolist()
            if isinstance(indices, np.ndarray):
                indices = indices.tolist()
            if isinstance(indptr, np.ndarray):
                indptr = indptr.tolist()
            array._csr_array.from_components(data, indices, indptr)
        else:
            # print("Using ", device.name)
            array.device.from_numpy_sparse(data, indices, indptr, array._csr_array)

        array.nnz = len(data)

        return array

    @property
    def to_numpy_array(self):
        """Convert the sparse ndarray to a numpy ndarray."""
        array = np.zeros(self._shape, dtype=np.float32)
        num_rows = self._shape[0]
        for i in range(num_rows):
            # breakpoint()
            start = self._csr_array.indptr[i]
            end = self._csr_array.indptr[i + 1]
            for j in range(start, end):
                array[i, self._csr_array.indices[j]] = self._csr_array.data[j]

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

    @staticmethod
    def create_random_sparse_matrix(
        shape: Tuple[int, int], density: float = 0.1
    ) -> "SparseNDArray":
        """
        Create a random sparse matrix with the given shape and density.
        Density is the fraction of non-zero elements in the matrix.
        """
        if not (0 <= density <= 1):
            raise ValueError("Density must be between 0 and 1.")

        dense_matrix = np.random.rand(*shape)
        mask = np.random.rand(*shape) < density
        sparse_matrix = np.multiply(dense_matrix, mask)

        return sparse_matrix

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
        # Create the C++ SparseArray backend
        sparse_array = self.device.SparseArray(self.nnz, self._shape[0], self._shape[1])
        sparse_array.from_components(self._csr_array.data, self._csr_array.indices, self._csr_array.indptr)
        return sparse_array

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
        raise NotImplementedError("Broadcasting is not implemented yet.")

    # TODO: What should be the type of "out" here? NDArray or SparseNDArray?
    # def ewise_or_scalar(self, other, ewise_func, scalar_func):
    #     """Run either an elementwise or scalar version of a function,
    #     depending on whether "other" is an SparseNDArray or scalar
    #     """
    #     # out = SparseNDArray.make(
    #     #     self._shape, self._data, self._indices, self._indptr, self._device
    #     # )

    #     # breakpoint()

    #     if isinstance(other, SparseNDArray):
    #         assert self.shape == other.shape, "operation needs two equal-sized arrays"
    #         out = SparseNDArray(self._shape, self._device)
    #         # out._shape = self._shape
    #         # out._device = self._device

    #         # Convert self and other into the expected C++ SparseArray type
    #         sparse_self = self.to_cpp_sparse_array
    #         sparse_other = other.to_cpp_sparse_array
    #         # out = out.to_cpp_sparse_array

    #         # Call the C++ elementwise function
    #         ewise_func(sparse_self, sparse_other, out._csr_array)

    #     elif isinstance(other, NDArray):
    #         out = full(shape=self._shape, fill_value=0, device=self._device)
    #         assert self.shape == other.shape, "operation needs two equal-sized arrays"
    #         # Convert self and other into the expected C++ SparseArray type
    #         sparse_self = self.to_cpp_sparse_array

    #         ewise_func(sparse_self, other, out._handle)

    #     # Case 2: Scalar operation
    #     else:
    #         out = SparseNDArray(self._shape, self._device)

    #         # Convert self into C++ SparseArray and call scalar_func
    #         sparse_self = self.to_cpp_sparse_array
    #         # out = out.to_cpp_sparse_array

    #         scalar_func(sparse_self, float(other), out._csr_array)

    #     return out

    def __add__(self, other):
        # out = None
        if isinstance(other, SparseNDArray) and isinstance(self, SparseNDArray):
            # return self.ewise_or_scalar(
            #     other,
            #     self.device.sparse_ewise_add_SSS,
            #     self.device.sparse_scalar_add,
            # )
            assert self.shape == other.shape, "operation needs two equal-sized arrays"
            out = SparseNDArray(self._shape, self._device)

            # Convert self and other into the expected C++ SparseArray type
            # sparse_self = self.to_cpp_sparse_array
            # sparse_other = other.to_cpp_sparse_array

            # Call the C++ elementwise function
            # self.device.sparse_ewise_add_SSS(sparse_self, sparse_other, out._csr_array)
            self.device.sparse_ewise_add_SSS(self._csr_array, other._csr_array, out._csr_array)

        elif isinstance(other, NDArray) and isinstance(self, SparseNDArray):
            # return self.ewise_or_scalar(
            #     other, self.device.sparse_ewise_add_SDD, self.device.sparse_scalar_add
            # )
            assert self.shape == other.shape, "operation needs two equal-sized arrays"
            out = full(shape=self._shape, fill_value=0, device=self._device)

            # Convert self and other into the expected C++ SparseArray type
            # sparse_self = self.to_cpp_sparse_array

            # self.device.sparse_ewise_add_SDD(sparse_self, other._handle, out._handle)
            self.device.sparse_ewise_add_SDD(self._csr_array, other._handle, out._handle)

        elif isinstance(other, SparseNDArray) and isinstance(self, NDArray):
            # return other.ewise_or_scalar(
            #     self, self.device.sparse_ewise_add_DSD, self.device.sparse_scalar_add
            # )
            assert self.shape == other.shape, "operation needs two equal-sized arrays"
            out = full(shape=self._shape, fill_value=0, device=self._device)

            # Convert self and other into the expected C++ SparseArray type
            # sparse_other = other.to_cpp_sparse_array

            # self.device.sparse_ewise_add_DSD(self, sparse_other, out._handle)
            self.device.sparse_ewise_add_DSD(self, other._csr_array, out._handle)

        elif isinstance(other, float) or isinstance(other, int):
            # return self.ewise_or_scalar(
            #     other, self.device.sparse_ewise_add_SDD, self.device.sparse_scalar_add
            # )
            # out = SparseNDArray.make(self._shape, self._csr_array.data, self._csr_array.indices, self._csr_array.indptr, self._device)
            out = full(shape=self._shape, fill_value=0, device=self._device)
            # Convert self into C++ SparseArray and call scalar_func
            # sparse_self = self.to_cpp_sparse_array
            # self.device.sparse_scalar_add(sparse_self, float(other), out._handle)
            self.device.sparse_scalar_add(self._csr_array, float(other), out._handle)

        else:
            raise ValueError("Unsupported type" + str(type(other)))

        return out

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        out = None
        if isinstance(other, SparseNDArray) and isinstance(self, SparseNDArray):
            # return self.ewise_or_scalar(
            #     other,
            #     self.device.sparse_ewise_mul_SSS,
            #     self.device.sparse_scalar_mul,
            # )
            assert self.shape == other.shape, "operation needs two equal-sized arrays"
            out = SparseNDArray(self._shape, self._device)
            # out._shape = self._shape
            # out._device = self._device

            # Convert self and other into the expected C++ SparseArray type
            # sparse_self = self.to_cpp_sparse_array
            # sparse_other = other.to_cpp_sparse_array
            # out = out.to_cpp_sparse_array

            # Call the C++ elementwise function
            # self.device.sparse_ewise_mul_SSS(sparse_self, sparse_other, out._csr_array)
            self.device.sparse_ewise_mul_SSS(self._csr_array, other._csr_array, out._csr_array)

        elif isinstance(other, NDArray) and isinstance(self, SparseNDArray):
            # return self.ewise_or_scalar(
            #     other, self.device.sparse_ewise_mul_SDD, self.device.sparse_scalar_mul
            # )
            assert self.shape == other.shape, "operation needs two equal-sized arrays"
            out = full(shape=self._shape, fill_value=0, device=self._device)

            # Convert self and other into the expected C++ SparseArray type
            # sparse_self = self.to_cpp_sparse_array

            # self.device.sparse_ewise_mul_SDD(sparse_self, other, out._handle)
            self.device.sparse_ewise_mul_SDD(self._csr_array, other._handle, out._handle)

        elif isinstance(other, SparseNDArray) and isinstance(self, NDArray):
            # return other.ewise_or_scalar(
            #     self, self.device.sparse_ewise_mul_DSD, self.device.sparse_scalar_mul
            # )
            assert self.shape == other.shape, "operation needs two equal-sized arrays"
            out = full(shape=self._shape, fill_value=0, device=self._device)

            # Convert self and other into the expected C++ SparseArray type
            # sparse_other = other.to_cpp_sparse_array

            # self.device.sparse_ewise_mul_DSD(self, sparse_other, out._handle)
            self.device.sparse_ewise_mul_DSD(self._handle, other._csr_array, out._handle)

        elif isinstance(other, float) or isinstance(other, int):
            # return self.ewise_or_scalar(
            #     other, self.device.sparse_ewise_mul_SDD, self.device.sparse_scalar_mul
            # )
            out = SparseNDArray.make(self._shape, self._csr_array.data, self._csr_array.indices, self._csr_array.indptr, self._device)

            # Convert self into C++ SparseArray and call scalar_func
            # sparse_self = self.to_cpp_sparse_array
            # self.device.sparse_scalar_mul(sparse_self, float(other), out._csr_array)
            self.device.sparse_scalar_mul(self._csr_array, float(other), out._csr_array)

        else:
            raise ValueError("Unsupported type" + str(type(other)))

        return out

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

        # breakpoint()
        if isinstance(other, SparseNDArray):
            # Handle sparse @ sparse case
            if other.shape[1] == 1:
                # Vector case
                out = SparseNDArray(output_shape, device=self.device)
                
                # sparse_self = self.to_cpp_sparse_array
                # sparse_other = other.to_cpp_sparse_array
                # self.device.sparse_mat_sparse_vec_mul(
                #     sparse_self, sparse_other, out._handle
                # )
                self.device.sparse_mat_sparse_vec_mul(
                    self._csr_array, other._csr_array, out._csr_array
                )
            else:
                # Matrix case
                out = SparseNDArray(output_shape, device=self.device)

                # sparse_self = self.to_cpp_sparse_array
                # sparse_other = other.to_cpp_sparse_array
                # self.device.sparse_mat_sparse_mat_mul_sparse(
                #     sparse_self, sparse_other, out._csr_array
                # )
                # breakpoint()
                self.device.sparse_mat_sparse_mat_mul_sparse(
                    self._csr_array, other._csr_array, out._csr_array
                )
        else:
            # Handle sparse @ dense case
            if other.shape[1] == 1:
                # Vector case
                out = full(shape=output_shape, fill_value=0, device=self.device)
                # sparse_self = self.to_cpp_sparse_array
                # sparse_other = other.to_cpp_sparse_array
                # self.device.sparse_mat_dense_vec_mul(
                #     sparse_self, sparse_other, out._handle
                # )
                self.device.sparse_mat_dense_vec_mul(
                    self._csr_array, other._handle, out._handle
                )
            else:
                # Matrix case
                out = full(shape=output_shape, fill_value=0, device=self.device)
                # sparse_self = self.to_cpp_sparse_array
                # sparse_other = other.to_cpp_sparse_array
                # self.device.sparse_mat_dense_mat_mul(
                #     sparse_self, sparse_other, out._handle
                # )
                self.device.sparse_mat_dense_mat_mul(
                    self._csr_array, other._handle, out._handle, output_shape[1]
                )

        return out


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
