"""Optimization module"""

import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        for p in self.params:
            if p.grad is None:
                continue
            if p not in self.u:
                self.u[p] = 0
            grad = ndl.Tensor(p.grad.data + self.weight_decay * p.data, device=p.data.device, requires_grad=False, dtype=p.data.dtype)
            self.u[p] = self.momentum * self.u[p] + (1 - self.momentum) * grad
            p.data = p.data - self.lr * self.u[p]
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        total_norm = 0
        for p in self.params:
            if p.grad is None:
                continue
            total_norm += np.linalg.norm(p.grad.detach().flatten()) ** 2
        total_norm = np.sqrt(total_norm)
        if total_norm > max_norm:
            for p in self.params:
                if p.grad is None:
                    continue
                p.grad = p.grad * (max_norm / total_norm)
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue
            if p not in self.m:
                self.m[p] = 0
                self.v[p] = 0
            grad = ndl.Tensor(p.grad.data + self.weight_decay * p.data, device=p.data.device, requires_grad=False, dtype=p.data.dtype)
            self.m[p] = self.beta1 * self.m[p] + (1 - self.beta1) * grad
            self.v[p] = self.beta2 * self.v[p] + (1 - self.beta2) * grad ** 2
            m_hat = self.m[p] / (1 - self.beta1 ** self.t)
            v_hat = self.v[p] / (1 - self.beta2 ** self.t)
            p.data = p.data - self.lr * m_hat / (v_hat ** 0.5 + self.eps)
        ### END YOUR SOLUTION