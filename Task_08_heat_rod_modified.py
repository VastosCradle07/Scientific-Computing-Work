from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

n = 7

A = 2.0 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)

b = np.zeros(n)
b[0] = 80.0
b[-1] = 30.0


def jacobi(A, b, tol=1e-8, max_iter=2000):
    D = np.diag(A)
    R = A - np.diagflat(D)
    x = np.zeros_like(b)
    errors = []
    for _ in range(max_iter):
        x_new = (b - R @ x) / D
        error = np.linalg.norm(x_new - x, ord=np.inf)
        errors.append(error)
        if error < tol:
            return x_new, errors
        x = x_new
    return x, errors


def gauss_seidel(A, b, tol=1e-8, max_iter=2000):
    x = np.zeros_like(b)
    errors = []
    for _ in range(max_iter):
        old = x.copy()
        for i in range(len(b)):
            x[i] = (
                b[i]
                - A[i, :i] @ x[:i]
                - A[i, i + 1:] @ old[i + 1:]
            ) / A[i, i]
        error = np.linalg.norm(x - old, ord=np.inf)
        errors.append(error)
        if error < tol:
            return x, errors
    return x, errors


x_j, err_j = jacobi(A, b)
x_gs, err_gs = gauss_seidel(A, b)
x_exact = np.linalg.solve(A, b)

print("Direct solution:", np.round(x_exact, 4))
print("Jacobi solution:", np.round(x_j, 4))
print("Gauss-Seidel solution:", np.round(x_gs, 4))
print("Jacobi iterations:", len(err_j))
print("Gauss-Seidel iterations:", len(err_gs))
print("Jacobi max error vs direct:", np.max(np.abs(x_j - x_exact)))
print("Gauss-Seidel max error vs direct:", np.max(np.abs(x_gs - x_exact)))

positions = np.arange(1, n + 1)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(positions, x_exact, marker="o")
axes[0].set_xlabel("Rod position")
axes[0].set_ylabel("Temperature (C)")
axes[0].set_title("Steady-state temperature along the rod")
axes[0].grid(True)
axes[1].semilogy(err_j, label="Jacobi")
axes[1].semilogy(err_gs, label="Gauss-Seidel")
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("Maximum change")
axes[1].set_title("Convergence")
axes[1].legend()
axes[1].grid(True)
fig.savefig(OUTPUT_DIR / "Task_08_heat_rod_modified.png", dpi=150, bbox_inches="tight")
plt.show()
