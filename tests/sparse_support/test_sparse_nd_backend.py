import sys
sys.path.append('./python')
import itertools
import numpy as np
import pytest
import mugrade
# import torch

import needle as ndl
from needle import backend_ndarray as nd
import timeit

np.random.seed(1)

def test_sparse_add():
    a = nd.sparse_ndarray.SparseNDArray.to_sparse(np.array([[2, 0, 0], [0, 0, 5]], dtype=np.float32), device=ndl.cpu())
    b = nd.sparse_ndarray.SparseNDArray.to_sparse(np.array([[1, 0, 0], [0, 2, 0]], dtype=np.float32), device=ndl.cpu())
    # breakpoint()
    c_ = nd.sparse_ndarray.sparse_add(a, b)
    c_np = c_.to_numpy_array
    # c_np = c.numpy()

    expected = np.array([[3, 0, 0], [0, 2, 5]], dtype=np.float32)
    assert np.allclose(c_np, expected)

def test_sparse_mul():
    a = nd.sparse_ndarray.SparseNDArray.to_sparse(np.array([[1, 0, 0], [0, 0, 1]], dtype=np.float32), device=ndl.cpu())
    b = nd.sparse_ndarray.SparseNDArray.to_sparse(np.array([[0, 0, 1], [0, 1, 0]], dtype=np.float32), device=ndl.cpu())

    c_ = nd.sparse_ndarray.sparse_mul(a, b)
    c_np = c_.to_numpy_array
    # c_np = c.numpy()

    expected = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float32)
    assert np.allclose(c_np, expected)

def test_sparse_add_scalar():
    a = nd.sparse_ndarray.SparseNDArray.to_sparse(np.array([[1, 0, 0], [0, 0, 1]], dtype=np.float32), device=ndl.cpu())
    # breakpoint()
    c_ = nd.sparse_ndarray.sparse_add(a, 1)
    # c_np = c_.to_numpy_array
    c_np = c_.numpy()

    expected = np.array([[2, 1, 1], [1, 1, 2]], dtype=np.float32)
    assert np.allclose(c_np, expected)

def test_sparse_mul_scalar():
    a = nd.sparse_ndarray.SparseNDArray.to_sparse(np.array([[1, 0, 0], [0, 0, 1]], dtype=np.float32), device=ndl.cpu())

    c_ = nd.sparse_ndarray.sparse_mul(a, 2)
    c_np = c_.to_numpy_array
    # c_np = c.numpy()

    expected = np.array([[2, 0, 0], [0, 0, 2]], dtype=np.float32)
    assert np.allclose(c_np, expected)

def test_sparse_sparse_matmul():
    def run_sparse_test():
        a = np.random.randint(low=500, high=1000)
        b = np.random.randint(low=500, high=1000)
        c_ = np.random.randint(low=500, high=1000)

        mat_1 = nd.sparse_ndarray.SparseNDArray.create_random_sparse_matrix((a, b))
        mat_2 = nd.sparse_ndarray.SparseNDArray.create_random_sparse_matrix((b, c_))

        mat_1_sparse = nd.sparse_ndarray.SparseNDArray.to_sparse(mat_1, device=ndl.cpu())
        # mat_2_sparse = nd.sparse_ndarray.SparseNDArray.to_sparse(mat_2, device=ndl.cpu())
        mat_2_dense = nd.ndarray.array(mat_2, device=ndl.cpu())

        # breakpoint()
        # mat_3_sparse = mat_1_sparse @ 
        mat_3 = mat_1_sparse @ mat_2_dense
        mat_3_np = mat_3.numpy()

        expected = np.matmul(mat_1, mat_2)

        assert np.allclose(mat_3_np, expected)

    avg_time = timeit.timeit(run_sparse_test, number=100) / 100
    print(f"Average time for sparse matmul: {avg_time:.6f} seconds")

def test_dense_matmul_for_sparse_matrices():
    def run_dense_test():
        a = np.random.randint(low=500, high=1000)
        b = np.random.randint(low=500, high=1000)
        c_ = np.random.randint(low=500, high=1000)

        mat_1 = nd.sparse_ndarray.SparseNDArray.create_random_sparse_matrix((a, b))
        mat_2 = nd.sparse_ndarray.SparseNDArray.create_random_sparse_matrix((b, c_))

        mat_1_dense = nd.ndarray.array(mat_1, device=ndl.cpu())
        mat_2_dense = nd.ndarray.array(mat_2, device=ndl.cpu())

        # breakpoint()
        mat_3_dense = mat_1_dense @ mat_2_dense
        mat_3_np = mat_3_dense.numpy()
        expected = np.matmul(mat_1, mat_2)
        assert np.allclose(mat_3_np, expected)

    avg_time = timeit.timeit(run_dense_test, number=100) / 100
    print(f"Average time for dense matmul: {avg_time:.6f} seconds")
