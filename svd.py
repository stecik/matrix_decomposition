import numpy as np
from typing import Tuple


def spectral_decomposition(
    A: np.array, tol: float = 1e-6, clean: bool = True
) -> Tuple[np.array, np.array, np.array]:
    """
    input: A is a symmetric matrix MxM
    output: (V, lam, V_T) V is orthogonal matrix mxm, lam is diagonal matrix mxm, V_T is the transpose of V
    """
    eigenval, V = np.linalg.eig(A)
    lam = np.diag(np.round(eigenval, decimals=6))
    if clean:
        # remove close to zero values, zero rows and columns
        lam[lam < tol] = 0
        lam = lam[~np.all(lam == 0, axis=1)]
        lam = lam[:, ~np.all(lam == 0, axis=0)]
        lam[lam == -0] = 0

    return V, lam, V.T


def svd(
    A: np.array, reduced=False, tol: float = 1e-6
) -> Tuple[np.array, np.array, np.array]:
    """
    input: A is any matrix MxN
    output: (U, sigma, V_T) U is orthogonal matrix MxM, sigma is diagonal matrix MxN, V_T is orthogonal matrix NxN
    """
    M, N = A.shape
    V, lam, V_T = spectral_decomposition(A.T @ A)
    print("lam", lam)
    r = np.linalg.matrix_rank(lam, tol=tol)
    print(r)
    sigma = np.zeros((M, N))
    S = np.sqrt(lam)
    sigma[:r, :r] = S
    print("sigma", sigma)
    V1 = V[:, :r]
    U1 = A @ V1 @ np.linalg.pinv(S)
    if reduced:
        return U1, S, V1.T

    # expand U1 to U (orthogonal matrix)
    U = gram_schmidt(U1)
    return U, sigma, V.T


def gram_schmidt(A: np.array) -> np.array:
    """
    input: A is a matrix MxN, where M >= N, A[:, :N] are orthonormal
    output: A is orthogonal matrix MxM
    """
    M, N = A.shape
    A = np.hstack((A, np.eye(M)[:, N:]))
    for i in range(N, M):
        v = A[:, i]
        for j in range(i):
            # u is normalized so projection onto span(u) is <v, u>u
            u = A[:, j]
            # subtract projection of v onto u from v to get perpendicular vector
            v = v - ((v.T @ u) * u)
        # normalize v
        v = v / np.linalg.norm(v)
        A[:, i] = v
    A[A == -0] = 0
    return A


A1 = np.array(
    [
        [6, -5, 2, 5],
        [2, 1, -4, 9],
        [3, -3, 1, 9],
    ]
)

A2 = np.array(
    [
        [1, 1],
        [1, 2],
        [1, 3],
        [1, 4],
    ]
)

A3 = np.array(
    [
        [4, 2],
        [3, 1],
    ]
)
A4 = np.array(
    [
        [1, 2],
        [3, 4],
        [5, 6],
    ]
)
A5 = np.array(
    [
        [1, 0, 2],
        [0, 3, 4],
    ]
)
A6 = np.array(
    [
        [5, 0, 0, 0],
        [0, 3, 0, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 0],
    ]
)
A7 = np.array(
    [
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [3, 4, 5, 6, 7],
        [4, 5, 6, 7, 8],
        [5, 6, 7, 8, 9],
    ]
)

A8 = np.random.rand(50, 100)
A9 = np.array([[i + j for j in range(70)] for i in range(30)])

A_list = [A1, A2, A3, A4, A6, A7, A8, A9]  # A5 is problematic
# A_list = [A5]
for i, A in enumerate(A_list):
    print(f"A{i}:")
    U, sigma, V_T = svd(A, reduced=False)
    print(U)
    print()
    print(sigma)
    print()
    print(V_T)
    print()

    print(U @ sigma @ V_T)
    assert np.allclose((U @ sigma @ V_T), A)
    assert np.allclose(np.round(U.T @ U, 4), np.eye(A.shape[0]))


# # Perform SVD
# U, sigma, V_T = np.linalg.svd(A5, full_matrices=True)

# # Construct the rectangular Sigma matrix
# Sigma = np.zeros((U.shape[1], V_T.shape[0]))  # Shape (2, 3) for A5
# np.fill_diagonal(Sigma, sigma)

# # Reconstruct the matrix
# reconstructed_A5 = U @ Sigma @ V_T

# print(np.round(reconstructed_A5, 6))
