from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND


class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        # print(type(Z))
        # Z = Tensor(Z)
        max_Z = array_api.max(Z, axis=-1, keepdims=True)
        max_Z_broadcast = array_api.max(Z, axis=-1)
        Log_softmax = (
            array_api.log(array_api.sum(array_api.exp(Z - max_Z), (-1)))
            + max_Z_broadcast
        )
        Log_softmax = Log_softmax.reshape((Z.shape[0], 1))
        # print("Z shape", Z.shape)
        # print("Log_softmax shape", Log_softmax.shape)
        final = Z - Log_softmax
        # print("final", final)
        return final
        # return Z - max_Z - logsoftmax(Z)
        # return Tensor(final)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        Z = node.inputs[0]

        # print("out_grad", out_grad.shape)
        # print("node", node.shape)
        # print("node.inputs[0]", node.inputs[0].shape)

        sum_grad = summation(out_grad, axes=-1).reshape((Z.shape[0], 1))
        final = sum_grad * exp(node)
        return out_grad - final

        ## END YOUR SOLUTION

# class LogSoftmax(TensorOp):
#     def compute(self, Z):
#         ### BEGIN YOUR SOLUTION
#         return Z - logsumexp(Z)
#         ### END YOUR SOLUTION

#     def gradient(self, out_grad, node):
#         ### BEGIN YOUR SOLUTION
#         z = node.inputs[0]
#         axes = list(range(len(z.shape)))
#         z = node.inputs[0]
#         z_max_dim = Tensor(z.get_outputs().max(axes, keepdims=True), device=z.device, requires_grad=False)
#         z_exp = exp(z + (-z_max_dim).broadcast_to(z.shape))
#         z_exp_sum = summation(z_exp, axes=axes)
#         grad_z_exp_sum = 1 / z_exp_sum
#         grad_z_exp_sum = grad_z_exp_sum.compute()
#         ori_shape = z.shape
#         sum_shape = range(len(z.shape)) if axes is None else axes
#         now_shape = list(ori_shape)
#         for i in sum_shape:
#             now_shape[i] = 1
#         return (1 - reshape(grad_z_exp_sum, now_shape).broadcast_to(ori_shape) * z_exp) * out_grad
#         ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        # print(type(Z))
        max_Z = Z.max(self.axes, keepdims=True)
        max_Z_broadcast = Z.max(self.axes)
        return (
            array_api.log(
                array_api.summation(array_api.exp(Z - max_Z.broadcast_to(Z.shape)), self.axes)
            )
            + max_Z_broadcast
        )
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        Z = node.inputs[0]
        Z_max = Tensor(Z.numpy().max(axis=self.axes), device=Z.device)

        # Determine the shape for reshaping and broadcasting
        Z_shape_for_reshape = list(Z.shape)
        if self.axes is not None:
            if isinstance(self.axes, int):
                self.axes = [self.axes]
            for axis in self.axes:
                Z_shape_for_reshape[axis] = 1
        else:
            for i in range(len(Z_shape_for_reshape)):
                Z_shape_for_reshape[i] = 1
        Z_shape_for_reshape = tuple(Z_shape_for_reshape)
        Z_shape_for_broadcast = Z.shape

        # Broadcast Z_max to match the shape of Z
        Z_max_reshaped_broadcasted = broadcast_to(
            reshape(Z_max, Z_shape_for_reshape), Z_shape_for_broadcast
        )
        Z_minus_Z_max = Z - Z_max_reshaped_broadcasted
        Z_exp = exp(Z_minus_Z_max)

        # Compute the sum of exponentials and broadcast
        Z_sum_exp = broadcast_to(
            reshape(summation(Z_exp, self.axes), Z_shape_for_reshape),
            Z_shape_for_broadcast,
        )

        # Final gradient calculation
        return multiply(
            broadcast_to(reshape(out_grad, Z_shape_for_reshape), Z_shape_for_broadcast),
            divide(Z_exp, Z_sum_exp),
        )
    
    # def gradient(self, out_grad, node):
    #     Z = node.inputs[0]

    #     # Determine the shape needed for broadcasting based on `self.axes`
    #     if self.axes is not None:
    #         new_shape = [1] * len(Z.shape)
    #         for axis in self.axes:
    #             new_shape[axis] = Z.shape[axis]
    #         # Reshape out_grad to include singleton dimensions along `self.axes`
    #         grad_new = reshape(out_grad, new_shape)
    #     else:
    #         grad_new = out_grad

    #     # Compute the softmax-like gradient term: exp(Z) / sum(exp(Z))
    #     Z_exp = exp(Z - node)  # `node` is already logsumexp(Z), so Z - node stabilizes this
    #     softmax_grad = Z_exp / summation(Z_exp, self.axes).reshape(new_shape)

    #     # Multiply `grad_new` with the computed softmax-like term
    #     return grad_new * softmax_grad

    # def gradient(self, out_grad, node):
    #     ### BEGIN YOUR SOLUTION
    #     # raise NotImplementedError()
    #     # node = output of logsumexp
    #     # out_grad = gradient of loss wrt node
    #     # we don't need to find max_Z
    #     # print("")
    #     # print("out_grad", out_grad.shape)
    #     # print("node", node.shape)
    #     # print("node.inputs[0]", node.inputs[0].shape)
    #     # print("self.axes", self.axes)

    #     Z = node.inputs[0]

    #     if self.axes:
    #         new_shape = [1] * len(Z.shape)
    #         s = set(self.axes)
    #         j = 0
    #         for i in range(len(Z.shape)):
    #             if i not in s:
    #                 new_shape[i] = node.shape[j]
    #                 j += 1
    #         # print("new_shape", new_shape)
    #         grad_new = out_grad.reshape(new_shape)
    #         node_new = node.reshape(new_shape)
    #     else:
    #         node_new = node
    #         grad_new = out_grad

    #     final = grad_new.broadcast_to(node_new.shape) * exp(node.inputs[0].broadcast_to(node_new.shape) - node_new)
    #     # print("final", final.shape)
    #     return final
    #     ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
