"""Sparse Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, SparseTensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *

# class SparseEWiseAdd(TensorOp):
    