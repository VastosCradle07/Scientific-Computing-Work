from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root_scalar

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

PRICE = 50.0
VARIABLE_COST = 20.0
FIXED_COST = 2000.0
CONGESTION = 0.05


def profit(q):
    return PRICE * q - (FIXED_COST + VARIABLE_COST * q + CONGESTION * q ** 2)


def g(q):
    return (FIXED_COST + CONGESTION * q ** 2) / (PRICE - VARIABLE_COST)


bisection = root_scalar(profit, bracket=[10, 200], method="bisect", xtol=1e-10)

q = 50.0
history = [q]

for _ in range(200):
    q_new = g(q)
    history.append(q_new)
    if abs(q_new - q) < 1e-10:
        break
    q = q_new

upper = root_scalar(profit, bracket=[300, 800], method="bisect", xtol=1e-10)

print(f"Break-even quantity (bisection): {bisection.root:.4f} units")
print(f"Bisection iterations: {bisection.iterations}")
print(f"Break-even quantity (fixed point): {history[-1]:.4f} units")
print("Fixed-point iterations:", len(history) - 1)
print(f"Second break-even quantity (profit turns negative again): {upper.root:.4f} units")
print(f"Profit at first break-even: {profit(bisection.root):.8f}")

quantities = np.linspace(0, 600, 400)

plt.plot(quantities, profit(quantities))
plt.axhline(0, linewidth=0.8)
plt.axvline(bisection.root, linestyle="--", label="First break-even")
plt.axvline(upper.root, linestyle=":", label="Second break-even")
plt.xlabel("Units sold")
plt.ylabel("Profit")
plt.title("Break-even analysis")
plt.legend()
plt.grid(True)
plt.savefig(OUTPUT_DIR / "Task_10_break_even_modified.png", dpi=150, bbox_inches="tight")
plt.show()
