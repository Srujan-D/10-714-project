import operator
import math
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
        self._shape = other._shape
        self._device = other._device
        self._data = other._data
        self._indices = other._indices
        self._indptr = other._indptr

    @staticmethod
    def make_sparse_from_numpy(other, device):
        """Convert a numpy ndarray to a sparse ndarray."""
        raise NotImplementedError()

    @staticmethod
    def make(shape, data, indices, indptr, device=None):
        """Create a sparse ndarray from the given data."""
        array = SparseNDArray.__new__(SparseNDArray)
        array._shape = tuple(shape)
        array._device = device
        array._data = data
        array._indices = indices
        array._indptr = indptr
        return array

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

    def __repr__(self):
        return f"SparseNDArray(shape={self._shape}, device={self._device}, dtype={self.dtype})"

    def __str__(self):
        return f"SparseNDArray(shape={self._shape}, device={self._device}, dtype={self.dtype})"

    def to(self, device):
        """Move the sparse ndarray to the given device."""
        if self._device == device:
            return self
        raise NotImplementedError()

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
            ewise_func(out, other)
        else:
            scalar_func(out, other)
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

        if isinstance(other, SparseNDArray):
            # self.device.sparse_matmul(self, other)
            raise NotImplementedError()
        else:
            # self.device.sparse_matmul_dense(self, other)
            raise NotImplementedError()
