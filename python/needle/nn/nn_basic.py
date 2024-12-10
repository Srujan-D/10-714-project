"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
        if bias:
            self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype).transpose())
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        out = X.matmul(self.weight)
        if self.bias:
            out += ops.broadcast_to(self.bias, (X.shape[0], self.weight.shape[1]))
        return out
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        # return ops.reshape(X, (X.shape[0], -1))
        shape = X.shape
        shape_prod = np.prod(shape)
        return ops.reshape(X, (X.shape[0], shape_prod // X.shape[0]))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.tanh(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


# class SoftmaxLoss(Module):
#     def forward(self, logits: Tensor, y: Tensor):
#         ### BEGIN YOUR SOLUTION
#         # raise NotImplementedError()
#         # print("logits", logits)
#         # print("y", y)
#         Z_y = init.one_hot(logits.shape[1], y, device=logits.device, dtype=logits.dtype)
#         # Z_y = logits * Z_y
#         # Z_y = ops.summation(Z_y, axes=(1,))
#         # print("Z_y", Z_y)
#         log_sum_exp = ops.logsumexp(logits, axes=(1,))
#         # print("log_sum_exp", log_sum_exp)
#         final = log_sum_exp - ops.summation(ops.multiply(logits, Z_y), axes=(1,))
#         final_sum = ops.summation(final, axes=(-1,))
#         return final_sum/y.shape[0]
#         ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        if len(logits.shape) == 1:
            logits = logits.reshape((1, logits.shape[0]))
            # y = y.reshape((1, y.shape[0]))
        
        num_examples, output_dim = logits.shape[0], logits.shape[1]
        y_onehot = init.one_hot(output_dim, y, device=logits.device, dtype=logits.dtype)
        if len(y_onehot.shape) == 1:
            y_onehot = y_onehot.reshape((1, y_onehot.shape[0]))
        # breakpoint()
        lhs = ops.logsumexp(logits, axes = 1)
        rhs = ops.summation(ops.multiply(logits, y_onehot), axes=1)
        return ops.divide_scalar(ops.summation(lhs-rhs), num_examples)
        ### END YOUR SOLUTION

class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)

        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        if self.training:
            mean = ops.summation(x, axes=(0,)).reshape((1, x.shape[1])) / x.shape[0]
            mean = ops.broadcast_to(mean, x.shape)
            var = ops.summation((x - mean) ** 2, axes=(0,)).reshape((1, x.shape[1])) / x.shape[0]
            var = ops.broadcast_to(var, x.shape)

            x_hat = (x - mean) / ops.power_scalar(var + self.eps, 0.5)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.sum(axes=0).detach()/x.shape[0]
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.sum(axes=0).detach()/x.shape[0]
        else:
            x_hat = (x - self.running_mean) / ops.power_scalar(self.running_var + self.eps, 0.5)
        
        broadcast_w = ops.broadcast_to(self.weight, x.shape)
        broadcast_b = ops.broadcast_to(self.bias, x.shape)
        
        return broadcast_w * x_hat + broadcast_b
        ### END YOUR SOLUTION

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        self.w = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.b = Parameter(init.zeros(dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        # print("x", x.shape)
        mean = ops.summation(x, axes=(1,)).reshape((x.shape[0], 1)) / x.shape[1]
        # print("mean", mean.shape, x.shape)
        mean = ops.broadcast_to(mean, x.shape)
        # print("broadcasted mean", mean.shape)
        var = ops.summation((x - mean) ** 2, axes=(1,)).reshape((x.shape[0], 1)) / x.shape[1]
        var = ops.broadcast_to(var, x.shape)

        x_hat = (x - mean) / ops.power_scalar(var + self.eps, 0.5)

        broadcast_w = ops.broadcast_to(self.w, x.shape)
        broadcast_b = ops.broadcast_to(self.b, x.shape)

        return broadcast_w * x_hat + broadcast_b
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        if self.training:
            mask = init.randb(*x.shape, p=1-self.p)
            return x * mask / (1 - self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return x + self.fn(x)
        ### END YOUR SOLUTION
