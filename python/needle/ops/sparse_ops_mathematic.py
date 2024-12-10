"""Sparse Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray, SparseNDArray
from ..autograd import Op, Tensor, SparseTensor, Value, SparseTensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

from .ops_mathematic import narrowcast, transpose

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class SparseEWiseAdd(SparseTensorOp):
    def compute(self, a: SparseNDArray, b: SparseNDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def sparse_add(a, b):
    return SparseEWiseAdd()(a, b)


# TODO: Check if this is needed, and think about the right way to implement it
class SparseEWiseAddTensor(SparseTensorOp):
    def compute(self, a: SparseNDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def sparse_add_tensor(a, b):
    return SparseEWiseAddTensor()(a, b)


class SparseAddScalar(SparseTensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: SparseNDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return SparseAddScalar(scalar)(a)


class SparseEWiseMul(SparseTensorOp):
    def compute(self, a: SparseNDArray, b: Union[SparseNDArray, NDArray]):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad * node.inputs[1], out_grad * node.inputs[0]


def sparse_mul(a, b):
    return SparseEWiseMul()(a, b)


# TODO: Check if this is needed, and think about the right way to implement it
class SparseEWiseMulTensor(SparseTensorOp):
    def compute(self, a: SparseNDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad * node.inputs[1], out_grad * node.inputs[0]


def sparse_mul(a, b):
    return SparseEWiseMulTensor()(a, b)


class SparseMulScalar(SparseTensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: SparseNDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return SparseMulScalar(scalar)(a)


class SparseBroadcastTo(SparseTensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a: SparseNDArray):
        return a.broadcast_to(self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad.sum(axis=0)


def broadcast_to(a, shape):
    return SparseBroadcastTo(shape)(a)

class SparseTranspose(SparseTensorOp):
    def compute(self, a: SparseNDArray):
        return a.transpose()

    def gradient(self, out_grad: Tensor, node: Tensor):
        return transpose(out_grad)
    
def transpose(a):
    return SparseTranspose()(a)


class SparseMatMul(SparseTensorOp):
    def compute(self, a: SparseNDArray, b: Union[SparseNDArray, NDArray]):
        # breakpoint()
        return a @ b

    def gradient(self, out_grad: Tensor, node: Tensor):
        # breakpoint()
        # raise NotImplementedError()
        a, b = node.inputs
        return (out_grad @ b.transpose(), a.transpose() @ out_grad)


def sparse_matmul(a, b):
    return SparseMatMul()(a, b)


class SparseNegate:
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return (self.compute(out_grad),)


def negate(a):
    return SparseNegate()(a)
