import timeit

import numpy as np
from scipy.linalg import lu_factor, lu_solve

A = np.array([
    [10.0, -4.0, 0.0],
    [-4.0, 12.0, -5.0],
    [0.0, -5.0, 9.0]
])

voltage_sets = [
    np.array([12.0, 0.0, -6.0]),
    np.array([12.0, 3.0, -6.0]),
    np.array([9.0, 0.0, -3.0]),
    np.array([0.0, 0.0, 0.0])
]

lu, piv = lu_factor(A)

for i, v in enumerate(voltage_sets, start=1):
    currents = lu_solve((lu, piv), v)
    residual = v - A @ currents
    print(f"Voltage set {i}: {v}")
    print("Mesh currents (A):", np.round(currents, 6))
    print("Residual norm:", np.linalg.norm(residual))
    print()

rng = np.random.default_rng(1)
n = 400
big_A = rng.random((n, n)) + n * np.eye(n)
right_hand_sides = rng.random((300, n))

big_lu, big_piv = lu_factor(big_A)


def repeated_solve():
    return [np.linalg.solve(big_A, rhs) for rhs in right_hand_sides]


def reused_lu():
    return [lu_solve((big_lu, big_piv), rhs) for rhs in right_hand_sides]


solve_time = timeit.timeit(repeated_solve, number=1)
lu_time = timeit.timeit(reused_lu, number=1)

print(f"300 solves with np.linalg.solve: {solve_time:.4f} s")
print(f"300 solves reusing LU factors: {lu_time:.4f} s")
print(f"Speed-up from reusing LU: {solve_time / lu_time:.1f}x")
