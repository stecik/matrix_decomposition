import numpy as np
from typing import Tuple


def asso_dbp(
    matrix: np.array, k: int, tau: float, w_plus: int, w_minus: int
) -> Tuple[np.array, np.array]:
    assoc_matrix = get_assoc_matrix(matrix, tau)
    S = np.zeros((matrix.shape[0], k), dtype=int)
    B = np.zeros((k, matrix.shape[1]), dtype=int)
    for i in range(k):
        max_cover = 0
        best_e = np.array([0 for _ in range(matrix.shape[0])])
        best_j = 0
        for j in range(matrix.shape[1]):
            mat_cover, e = cover(assoc_matrix[j], matrix, S, B, w_plus, w_minus)
            if mat_cover > max_cover:
                max_cover = mat_cover
                best_e = e
                best_j = j
        # print(max_cover)
        # print(best_e)
        # print(assoc_matrix[best_j])
        # print()
        S[:, i] = best_e
        B[i, :] = assoc_matrix[best_j]
        # print(S)
        # print(B)
        # print(bool_mat_mul(S, B))
        # print()
        # print("__________________________________________________________")
    return (S, B)


def asso_afp(
    matrix: np.array, acc: float, tau: float, w_plus: int, w_minus: int
) -> Tuple[np.array, np.array]:
    assoc_matrix = get_assoc_matrix(matrix, tau)
    S = np.zeros((matrix.shape[0], 1), dtype=int)
    B = np.zeros((1, matrix.shape[1]), dtype=int)
    i = 0
    fact_cov = factor_coverage(matrix, S, B)
    old_cover = fact_cov - 1
    while fact_cov < acc and old_cover < fact_cov:
        old_cover = fact_cov
        max_cover = 0
        best_e = np.array([0 for _ in range(matrix.shape[0])])
        best_j = 0
        for j in range(matrix.shape[1]):
            mat_cover, e = cover(assoc_matrix[j], matrix, S, B, w_plus, w_minus)
            if mat_cover > max_cover:
                max_cover = mat_cover
                best_e = e
                best_j = j
        # print(max_cover)
        # print(best_e)
        # print(assoc_matrix[best_j])
        # print()
        S[:, i] = best_e
        B[i, :] = assoc_matrix[best_j]
        i += 1
        S = np.hstack([S, np.zeros((matrix.shape[0], 1), dtype=int)])
        B = np.vstack([B, np.zeros((1, matrix.shape[1]), dtype=int)])
        fact_cov = factor_coverage(matrix, S, B)
    return (S, B, fact_cov)


def factor_coverage(matrix, S, B) -> float:
    SB = bool_mat_mul(S, B)
    SB[matrix == 0] = 0
    return np.sum(SB) / np.sum(matrix)


def cover(
    asso_mat_row: np.array,
    mat: np.array,
    S: np.array,
    B: np.array,
    w_plus: int,
    w_minus: int,
) -> Tuple[int, np.array]:
    SB = bool_mat_mul(S, B)
    uncovered = np.ones(mat.shape, dtype=int)
    uncovered[SB == 1] = 0  # 1 in SB => 0 in uncovered
    e = np.array([0 for _ in range(mat.shape[0])])
    mat_cover = 0
    for i in range(mat.shape[0]):
        pc_plus = 0
        pc_minus = 0
        for j in range(mat.shape[1]):
            pc_plus += asso_mat_row[j] * uncovered[i][j] * mat[i][j]
            pc_minus += asso_mat_row[j] * uncovered[i][j] * (1 - mat[i][j])
        partial_cover = pc_plus * w_plus - pc_minus * w_minus
        if partial_cover < 0:
            partial_cover = 0
        else:
            e[i] = 1
        mat_cover += partial_cover
    return (mat_cover, e)


def get_assoc_matrix(matrix: np.array, tau: float) -> np.array:
    assoc_martrix = np.eye(matrix.shape[1], matrix.shape[1], dtype=int)
    for i in range(assoc_martrix.shape[0]):
        for j in range(assoc_martrix.shape[1]):
            if not i == j:
                count_both = np.sum((matrix[:, i] == 1) & (matrix[:, j] == 1))
                count_atr1 = np.sum(matrix[:, i] == 1)
                if confidence(count_atr1, count_both) >= tau:
                    assoc_martrix[i, j] = 1
    return assoc_martrix


def confidence(count_atr1: int, count_both: int) -> float:
    return count_both / count_atr1


def bool_mat_mul(A: np.array, B: np.array) -> np.array:
    return np.dot(A.astype(int), B.astype(int)).clip(0, 1)


C = np.array(
    [
        [1, 1, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 1, 0],
    ]
)

# S, B = asso_dbp(C, k=2, tau=0.5, w_plus=1, w_minus=1)
# print(S)
# print()
# print(B)
# print()
# print(C)
# print()
# print(bool_mat_mul(S, B))

S, B, acc = asso_afp(C, acc=0.95, tau=0.5, w_plus=1, w_minus=1)
print(f"Accuracy: {acc}")
print(S)
print()
print(B)
print()
print(C)
print()
print(bool_mat_mul(S, B))


C = np.array(
    [
        [1, 0, 1, 0, 1, 1],
        [0, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1],
        [0, 1, 1, 1, 0, 1],
    ]
)
S, B, acc = asso_afp(C, acc=0.95, tau=0.5, w_plus=1, w_minus=1)
print(f"Accuracy: {acc}")
print(S)
print()
print(B)
print()
print(C)
print()
print(bool_mat_mul(S, B))
