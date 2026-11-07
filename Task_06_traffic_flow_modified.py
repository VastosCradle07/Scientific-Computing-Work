import numpy as np

A = np.array([
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0]
])

b = np.array([800.0, 700.0, 500.0])

solution = np.linalg.solve(A, b)
residual = b - A @ solution

print("Road 1 flow (vehicles/hour):", solution[0])
print("Road 2 flow (vehicles/hour):", solution[1])
print("Road 3 flow (vehicles/hour):", solution[2])
print("Verification A @ solution:", A @ solution)
print("Residual norm:", np.linalg.norm(residual))
print("Determinant:", np.linalg.det(A))
print("Condition number:", np.linalg.cond(A))
