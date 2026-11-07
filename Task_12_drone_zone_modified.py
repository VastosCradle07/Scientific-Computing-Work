import numpy as np
from scipy.optimize import root


def intersections(z):
    x, y = z
    return [
        (x - 2) ** 2 + (y - 1) ** 2 - 9,
        y - (0.5 * x - 1)
    ]


for guess in ([5.0, 1.5], [-0.2, -1.0]):
    solution = root(intersections, guess)
    print("Initial guess:", guess)
    print("Path enters or leaves the zone at:", np.round(solution.x, 4))
    print("Residuals:", [float(r) for r in intersections(solution.x)])
    print("Converged:", solution.success)
    print()

coefficients = [-4.9, 20.0, 2.0]
roots = np.roots(coefficients)
landing_time = max(r.real for r in roots if abs(r.imag) < 1e-12)

print("Projectile height roots (s):", np.round(roots, 4))
print(f"Landing time: {landing_time:.4f} s")
print(f"Height at landing time: {np.polyval(coefficients, landing_time):.2e} m")
