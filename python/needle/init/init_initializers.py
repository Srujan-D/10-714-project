import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    # raise NotImplementedError()

    a = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return rand(*(fan_in, fan_out), low=-a, high=a, **kwargs)
    ### END YOUR SOLUTION


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    # raise NotImplementedError()
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return randn(*(fan_in, fan_out), mean=0.0, std=std, **kwargs)
    ### END YOUR SOLUTION

def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", shape=None, **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    # raise NotImplementedError()
    bound = math.sqrt(6.0 / fan_in)
    if shape is not None:
        return rand(*(shape), low=-bound, high=bound, **kwargs)
    return rand(*(fan_in, fan_out), low=-bound, high=bound, **kwargs)
    ### END YOUR SOLUTION



def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    # raise NotImplementedError()
    std = math.sqrt(2.0 / fan_in)
    return randn(*(fan_in, fan_out), mean=0.0, std=std, **kwargs)
    ### END YOUR SOLUTION