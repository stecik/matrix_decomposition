import numpy as np
from typing import Tuple


np.set_printoptions(suppress=True, precision=4)


def householder(a: np.array) -> np.array:
    return np.eye(a.shape[0]) - 2 * np.outer(a, a.T) / np.dot(a.T, a)


def qr(A: np.array) -> Tuple[np.array, np.array]:
    m, n = A.shape
    e = np.eye(m)
    Q = np.eye(m)
    R = A.copy()
    for i in range(min(m, n)):
        x = R[i:m, i]
        if not np.allclose(x, np.linalg.norm(x) * e[i:, i]):
            x = x - np.linalg.norm(x) * e[i:, i]
            H = householder(x)
            H_full = np.eye(m)
            H_full[i:, i:] = H
            R = H_full @ R
            Q = Q @ H_full
    return (Q, R)


A1 = np.array(
    [
        [6, -5, 2],
        [2, 1, -4],
        [3, -3, 1],
    ]
)
A2 = np.array(
    [
        [2, 2, -1, 1],
        [4, 3, -1, 2],
        [8, 5, -3, 4],
        [3, 3, -2, 2],
    ]
)
A3 = np.array(
    [
        [2, -1, 3, 4, 1],
        [1, 5, 2, -1, 3],
        [3, 1, 4, 2, -2],
        [5, -2, 1, 3, 4],
        [-1, 2, 5, 1, 3],
    ]
)
A4 = np.array(
    [
        [3, 2, -1, 5, 4, 1],
        [1, 3, 4, 2, -1, 3],
        [-2, 5, 1, 3, 2, 4],
        [4, 1, -3, 5, -2, 2],
        [5, 3, 2, -1, 1, 6],
        [2, -2, 1, 4, 5, -3],
    ]
)
A5 = np.array(
    [
        [4, 3, -2, 5, 1, 6, 2],
        [1, -3, 5, 4, -2, 3, 1],
        [2, 5, -1, 3, 4, -2, 6],
        [6, 1, 3, -5, 2, 4, -1],
        [3, 4, 5, -2, 1, 2, 6],
        [5, -1, 2, 3, 6, 1, 4],
        [-2, 6, 4, 1, 3, -3, 5],
    ]
)

A_list = [A1, A2, A3, A4, A5]
for A in A_list:
    Q1, R1 = qr(A)
    print(Q1)
    print(R1)
    print()
    Q2, R2 = np.linalg.qr(A)
    print(Q2)
    print(R2)
    print()
    assert np.allclose(Q1 @ R1, Q2 @ R2, atol=1e-10)
