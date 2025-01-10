import numpy as np
from typing import Tuple


def nmf(A: np.array, k: int, max_iter: int) -> Tuple[np.array, np.array]:
    """
    input: A (m x n) non-negative matrix, k number of components, max_iter number of iterations
    output: W (m x k) and H (k x n) non-negative matrices
    """
    W = np.random.rand(A.shape[0], k)
    H = np.random.rand(k, A.shape[1])
    for i in range(max_iter):
        W = W * (A @ H.T) / (W @ H @ H.T)
        W = W / W.sum(axis=0)
        H = H * (W.T @ A) / (W.T @ W @ H)
        err = np.linalg.norm(A - W @ H)
        print(f"Iteration: {i}, Error: {err}")
    return (W, H)


A = np.array(
    [
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [3, 4, 5, 6, 7],
        [4, 5, 6, 7, 8],
        [5, 6, 7, 8, 9],
    ]
)

W, H = nmf(A, 3, 500)
