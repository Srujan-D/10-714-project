import sys
sys.path.append('./python')
import itertools
import numpy as np
import pytest
import mugrade
import torch

import needle as ndl
from needle import backend_ndarray as nd

np.random.seed(1)

def test_sparse_add():
    a = nd.sparse_ndarray.SparseNDArray.to_sparse(np.array([[1, 0, 0], [0, 0, 1]], dtype=np.float32), device=ndl.cpu())
    b = nd.sparse_ndarray.SparseNDArray.to_sparse(np.array([[0, 0, 1], [0, 1, 0]], dtype=np.float32), device=ndl.cpu())

    c = nd.sparse_ndarray.sparse_add(a, b)
    # c_np = c.to_numpy_array()
    c_np = c.numpy()

    expected = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.float32)
    assert np.allclose(c_np, expected)

def test_sparse_mul():
    a = nd.sparse_ndarray.SparseNDArray.to_sparse(np.array([[1, 0, 0], [0, 0, 1]], dtype=np.float32), device=ndl.cpu())
    b = nd.sparse_ndarray.SparseNDArray.to_sparse(np.array([[0, 0, 1], [0, 1, 0]], dtype=np.float32), device=ndl.cpu())

    c = nd.sparse_ndarray.sparse_mul(a, b)
    # c_np = c.to_numpy_array()
    c_np = c.numpy()

    expected = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float32)
    assert np.allclose(c_np, expected)

def test_sparse_add_scalar():
    a = nd.sparse_ndarray.SparseNDArray.to_sparse(np.array([[1, 0, 0], [0, 0, 1]], dtype=np.float32), device=ndl.cpu())

    c = nd.sparse_ndarray.add_scalar(a, 1)
    # c_np = c.to_numpy_array()
    c_np = c.numpy()

    expected = np.array([[2, 1, 1], [1, 1, 2]], dtype=np.float32)
    assert np.allclose(c_np, expected)

def test_sparse_mul_scalar():
    a = nd.sparse_ndarray.SparseNDArray.to_sparse(np.array([[1, 0, 0], [0, 0, 1]], dtype=np.float32), device=ndl.cpu())

    c = nd.sparse_ndarray.mul_scalar(a, 2)
    # c_np = c.to_numpy_array()
    c_np = c.numpy()

    expected = np.array([[2, 0, 0], [0, 0, 2]], dtype=np.float32)
    assert np.allclose(c_np, expected)

def test_sparse_sparse_matmul():
    a = np.random.randint(low=1, high=20)
    b = np.random.randint(low=1, high=20)
    c = np.random.randint(low=1, high=20)

    mat_1 = nd.sparse_ndarray.SparseNDArray.create_random_sparse_matrix((a, b))
    mat_2 = nd.sparse_ndarray.SparseNDArray.create_random_sparse_matrix((b, c))

    mat_1_sparse = nd.sparse_ndarray.SparseNDArray.to_sparse(mat_1, device=ndl.cpu())
    mat_2_sparse = nd.sparse_ndarray.SparseNDArray.to_sparse(mat_2, device=ndl.cpu())

    # breakpoint()
    mat_3_sparse = mat_1_sparse @ mat_2_sparse

    # mat_3_np = mat_3_sparse.to_numpy_array()
    mat_3_np = mat_3_sparse.numpy()

    expected = np.matmul(mat_1, mat_2)

    assert np.allclose(mat_3_np, expected)

