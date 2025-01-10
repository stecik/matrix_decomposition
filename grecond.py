import numpy as np
from typing import Set, Tuple
import itertools


def grecond_afp(matrix: np.array, acc: float) -> Set:
    ones_count = np.sum(matrix)
    uncovered = set(
        [
            (i, j)
            for i in range(matrix.shape[0])
            for j in range(matrix.shape[1])
            if matrix[i, j] == 1
        ]
    )
    formal_concepts = []
    j_set = set([j for j in range(matrix.shape[1])])
    current_acc = (ones_count - len(uncovered)) / ones_count
    while current_acc < acc:
        D = set()
        V = 0
        improved = True
        while improved:
            improved = False
            index = 0
            for j in j_set - D:
                size = len(circle_plus_operation(D, j, uncovered, matrix))
                if size > V:
                    improved = True
                    V = size
                    index = j
            if improved:
                D = up(down(D.union({index}), matrix), matrix)
                V = len(
                    set(itertools.product(down(D, matrix), D)).intersection(uncovered)
                )
        C = down(D, matrix)
        formal_concepts.append((C, D))
        uncovered = uncovered - set(itertools.product(C, D))
        current_acc = (ones_count - len(uncovered)) / ones_count
    return (formal_concepts, current_acc)


def grecond_dbp(matrix: np.array, k: int) -> Set:
    uncovered = set(
        [
            (i, j)
            for i in range(matrix.shape[0])
            for j in range(matrix.shape[1])
            if matrix[i, j] == 1
        ]
    )
    formal_concepts = []
    j_set = set([j for j in range(matrix.shape[1])])
    while len(formal_concepts) < k and len(uncovered) > 0:
        D = set()
        V = 0
        improved = True
        while improved:
            improved = False
            index = 0
            for j in j_set - D:
                size = len(circle_plus_operation(D, j, uncovered, matrix))
                if size > V:
                    improved = True
                    V = size
                    index = j
            if improved:
                D = up(down(D.union({index}), matrix), matrix)
                V = len(
                    set(itertools.product(down(D, matrix), D)).intersection(uncovered)
                )
        C = down(D, matrix)
        formal_concepts.append((C, D))
        uncovered = uncovered - set(itertools.product(C, D))
    return formal_concepts


def grecond(matrix: np.array) -> Set:
    uncovered = set(
        [
            (i, j)
            for i in range(matrix.shape[0])
            for j in range(matrix.shape[1])
            if matrix[i, j] == 1
        ]
    )
    formal_concepts = []
    j_set = set([j for j in range(matrix.shape[1])])
    while len(uncovered) > 0:
        D = set()
        V = 0
        improved = True
        while improved:
            improved = False
            index = 0
            for j in j_set - D:
                size = len(circle_plus_operation(D, j, uncovered, matrix))
                if size > V:
                    improved = True
                    V = size
                    index = j
            if improved:
                D = up(down(D.union({index}), matrix), matrix)
                V = len(
                    set(itertools.product(down(D, matrix), D)).intersection(uncovered)
                )
        C = down(D, matrix)
        formal_concepts.append((C, D))
        uncovered = uncovered - set(itertools.product(C, D))
    return formal_concepts


def circle_plus_operation(D: Set, y: int, U: Set, matrix: np.array) -> Set:
    """D ⊕ y = ((D ∪ {y})↓ × (D ∪ {y})↓↑) ∩ U"""
    D_down = down((D.union({y})), matrix)
    D_down_up = up(D_down, matrix)
    cartesian_product = set(itertools.product(D_down, D_down_up))
    return cartesian_product.intersection(U)


def down(attributes: Set, matrix: np.array) -> Set:
    """D↓ = {x ∈ X | for each y ∈ D: <x, y> ∈ I}"""
    objects = set()
    for i in range(matrix.shape[0]):
        valid = True
        for attribute in attributes:
            if matrix[i, attribute] != 1:
                valid = False
                break
        if valid:
            objects.add(i)
    return objects


def up(objects: Set, matrix: np.array) -> Set:
    """C↑ = {y ∈ Y | for each x ∈ C: <x, y> ∈ I}"""
    attributes = set([i for i in range(matrix.shape[1])])
    for object in objects:
        to_remove = set()
        for attribute in attributes:
            if matrix[object, attribute] == 0:
                to_remove.add(attribute)
        attributes = attributes - to_remove
    return attributes


def fc2matrices(formal_concepts: Set, shape: tuple) -> Tuple[np.array, np.array]:
    """Converts formal concepts to matrices A and B"""
    A = np.zeros((shape[0], len(formal_concepts)), dtype=int)
    B = np.zeros((len(formal_concepts), shape[1]), dtype=int)
    for i in range(len(formal_concepts)):
        coordinates = set(
            itertools.product(formal_concepts[i][0], formal_concepts[i][1])
        )
        for coordinate in coordinates:
            A[coordinate[0], i] = 1
            B[i, coordinate[1]] = 1
    return (A, B)


def bool_mat_mul(A: np.array, B: np.array) -> np.array:
    return np.dot(A.astype(int), B.astype(int)).clip(0, 1)


A = np.array(
    [
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ]
)

A = np.array(
    [
        [1, 0, 1, 0, 1, 1],
        [0, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1],
        [0, 1, 1, 1, 0, 1],
    ]
)

formal_concepts, acc = grecond_afp(A, 1)
print(formal_concepts)
print(acc)
print()

# print(grecond_dbp(A, 2))

A, B = fc2matrices(formal_concepts, A.shape)
print(A)
print()
print(B)
print()
print(bool_mat_mul(A, B))
