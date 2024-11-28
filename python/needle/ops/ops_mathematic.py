"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return a**b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        a, b = node.inputs
        return multiply(multiply(b, self.compute(a, b - 1)), out_grad), multiply(
            multiply(log(a), power(a, b)), out_grad
        )
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        x = node.inputs[0]
        return multiply(
            power_scalar(x, self.scalar - 1),
            mul_scalar(out_grad, self.scalar)
        )
        # return multiply(
        #     power_scalar(node.inputs[0], self.scalar - 1),
        #     mul_scalar(out_grad, self.scalar),
        # )
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.divide(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        a, b = node.inputs
        
        # Gradient with respect to `a`: out_grad * (1 / b)
        grad_a = self.compute(out_grad, b)
        
        # Gradient with respect to `b`: out_grad * (-a / b^2)
        grad_b = self.compute(
            multiply(broadcast_to(negate(a), out_grad.shape), out_grad), 
            power_scalar(b, 2)
        )
        
        return grad_a, grad_b
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return (out_grad / self.scalar,)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        # if self.axes is None:
        #     return array_api.swapaxes(a, -1, -2)
        # else:
        #     return array_api.swapaxes(a, self.axes[0], self.axes[1])
        index = list(range(len(a.shape)))
        if self.axes is None:
            index[-1], index[-2] = index[-2], index[-1]
        else:
            axis1, axis2 = self.axes[0], self.axes[1]
            index[axis1], index[axis2] = index[axis2], index[axis1]
        return a.permute(tuple(index))
        # return array_api.transpose(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return (transpose(out_grad, self.axes),)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return a.compact().reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return (reshape(out_grad, node.inputs[0].shape),)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


def narrowcast(broadcasted, input_shape):
    out_shape = broadcasted.shape
    axis_to_sum = []

    # Calculate the offset in dimensions between broadcasted and input shapes
    offset = len(out_shape) - len(input_shape)

    # Identify the axes to sum over
    for i in range(len(out_shape)):
        # If the dimension exists in input_shape, and it matches, skip
        if i >= offset and input_shape[i - offset] == out_shape[i]:
            continue
        # Otherwise, it's a broadcasted dimension that we need to sum over
        axis_to_sum.append(i)

    # Sum over the broadcasted axes
    if axis_to_sum:
        broadcasted = summation(broadcasted, axes=tuple(axis_to_sum))

    # Reshape the result to match the input shape
    return reshape(broadcasted, input_shape)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # ### BEGIN YOUR SOLUTION
        # # raise NotImplementedError()
        return narrowcast(out_grad, node.inputs[0].shape)
        # ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        if self.axes is None:
            return array_api.summation(a)
        elif isinstance(self.axes, int) or (isinstance(self.axes, (list, tuple)) and len(self.axes) == 1):
            return array_api.summation(a, axis=self.axes)
        else:
            for axis in reversed(sorted(self.axes)):
                a = array_api.summation(a, axis=axis)
            return a
        ### END YOUR SOLUTION
    def gradient(self, out_grad, node):
        # Get the shape of the original input
        input_shape = node.inputs[0].shape
        
        # Determine the axes over which summation was done
        axes = range(len(input_shape)) if self.axes is None else self.axes
        if isinstance(axes, int):
            axes = [axes]
        
        # Set dimensions in the output gradient shape to 1 for broadcasting
        new_shape = list(input_shape)
        for axis in axes:
            new_shape[axis] = 1
        
        # Reshape `out_grad` to the new shape
        reshaped_grad = reshape(out_grad, new_shape)
        
        # Broadcast `reshaped_grad` to match the input shape
        return (broadcast_to(reshaped_grad, input_shape),)
    # def gradient(self, out_grad, node):
    #     ### BEGIN YOUR SOLUTION
    #     # raise NotImplementedError()
    #     # broadcast the gradient to the input shape and divide by number of axes summed over
    #     input_shape = list(node.inputs[0].shape)
    #     axes = range(len(input_shape)) if self.axes is None else self.axes
    #     if isinstance(axes, Number):
    #         axes = [axes]
    #     # try:
    #     for axis in axes:
    #         new_shape[axis] = 1
    #     # except:
    #     #     print('axes', axes)
    #     #     print('new_shape', new_shape)
    #     #     print('node.inputs[0].shape', node.inputs[0].shape)
    #     #     for axis in axes:
    #     #         new_shape[axis] = 1
    #     out_grad = reshape(out_grad, new_shape)
    #     return (broadcast_to(out_grad, node.inputs[0].shape),)
    #     ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        a, b = node.inputs
        # print(a.shape, b.shape, out_grad.shape)
        # print('-'*10)
        # print(out_grad.shape, transpose(b).shape)
        # print('+'*10)
        # print(transpose(a).shape, out_grad.shape)
        # print('='*10)
        a1, a2 = matmul(out_grad, transpose(b)), matmul(transpose(a), out_grad)
        # print('a1 ', a1.shape, ' a2', a2.shape)
        if a1.shape != a.shape:
            a1 = narrowcast(a1, a.shape)
        if a2.shape != b.shape:
            a2 = narrowcast(a2, b.shape)
        # print('a1 ', a1.shape, ' a2', a2.shape)

        return a1, a2

        # return matmul(out_grad, transpose(b)), matmul(transpose(a), out_grad)
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.negative(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return (self.compute(out_grad),)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return divide(out_grad, node.inputs[0])
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return multiply(out_grad, exp(node.inputs[0]))
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0].realize_cached_data()
        return out_grad * Tensor(a > 0, device=out_grad.device)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        # broadcast out_grad to the shape of the input
        x = node.inputs[0]
        y = broadcast_to(out_grad, x.shape)
        return out_grad - multiply(y, power_scalar(tanh(x), 2))
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        new_shape = list(args[0].shape)
        new_shape.insert(self.axis, len(args))
        new_array = array_api.empty(new_shape, device=args[0].device)
        
        indices = [slice(None)] * len(new_shape)
        for i, arg in enumerate(args):
            indices[self.axis] = i
            new_array[tuple(indices)] = arg
        return new_array
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        # Get number of splits from the specified axis dimension
        split_size = A.shape[self.axis]
        
        # Create the basic slic_ing tuple for all dimensions
        slices = []
        for i, dim in enumerate(A.shape):
            if i != self.axis:
                slices.append(slice(0, dim))
        
        result = []
        for i in range(split_size):
            # Insert the current index for the split axis
            curr_slices = slices.copy()
            curr_slices.insert(self.axis, slice(i, i+1))
            
            # Get the slice and remove the split dimension through reshape
            new_shape = list(A.shape)
            del new_shape[self.axis]
            split_tensor = A[tuple(curr_slices)].compact().reshape(new_shape)
            
            result.append(split_tensor)
            
        return tuple(result)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.flip(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        new_shape = list(a.shape)
        for axis in self.axes:
            if axis >= len(new_shape):
                continue
            new_shape[axis] = a.shape[axis] * (self.dilation+1)
        new_array = array_api.full(new_shape, 0, device=a.device)
        indices = [slice(0, shape) for shape in new_shape]
        for ax in self.axes:
            if ax >= len(new_shape):
                continue
            indices[ax] = slice(None, None, self.dilation+1)
        new_array[tuple(indices)] = a
        return new_array
        
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        new_shape = list(a.shape)
        for axis in self.axes:
            if axis >= len(new_shape):
                continue
            new_shape[axis] = a.shape[axis] // (self.dilation+1)
        new_array = array_api.full(new_shape, 0, device=a.device)
        indices = [slice(0, shape) for shape in new_shape]
        for ax in self.axes:
            if ax >= len(new_shape):
                continue
            indices[ax] = slice(None, None, self.dilation+1)
        new_array = a[tuple(indices)]
        return new_array
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        if self.padding != 0:
            A = array_api.pad(A, [(0,0)]+[(self.padding, self.padding)]*2+[(0,0)])
        Ns, Hs, Ws, C_in = A.shape
        Kh, Kw, B_cin, C_out = B.shape

        assert C_in == B_cin, "Input and filter channels must match"

        Sn, Sh, Sw, Sc = A.strides

        inner_dim = Kh * Kw * C_in
        out_H, out_W = (Hs - Kh + 1)//self.stride, (Ws - Kw + 1)//self.stride
        new_shape = (Ns, out_H, out_W, Kh, Kw, C_in)
        new_strides = (Sn, Sh*self.stride, Sw*self.stride, Sh, Sw, Sc)

        A_strided = A.as_strided(new_shape, new_strides).compact().reshape((Ns*out_H*out_W, inner_dim))
        B = B.compact().reshape((inner_dim, C_out))
        A_dot_B = A_strided @ B
        return A_dot_B.compact().reshape((Ns, out_H, out_W, C_out))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        X, W = node.inputs
        K = W.shape[0]

        if self.stride > 1:
            out_grad = dilate(out_grad, (1, 2), self.stride - 1)
        W_transpose = transpose(flip(W, (0, 1)), (2, 3))
        X_grad = conv(out_grad, W_transpose, padding=K-1-self.padding)

        X_permute = transpose(X, (0, 3))
        out_grad_permute = transpose(transpose(out_grad, (0, 1)), (1, 2))
        W_grad = conv(X_permute, out_grad_permute, padding=self.padding)
        W_grad = transpose(transpose(W_grad, (0, 1)), (1, 2))

        return X_grad, W_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


