from scipy.optimize import minimize

PROFIT_CHAIR = 30.0
PROFIT_TABLE = 50.0
WOOD_AVAILABLE = 100.0
LABOUR_AVAILABLE = 90.0


def negative_profit(z):
    chairs, tables = z
    return -(PROFIT_CHAIR * chairs + PROFIT_TABLE * tables)


constraints = [
    {"type": "ineq", "fun": lambda z: WOOD_AVAILABLE - (2 * z[0] + 4 * z[1])},
    {"type": "ineq", "fun": lambda z: LABOUR_AVAILABLE - (3 * z[0] + 2 * z[1])}
]

bounds = [(0, None), (0, None)]

solution = minimize(
    negative_profit,
    x0=[1.0, 1.0],
    method="SLSQP",
    bounds=bounds,
    constraints=constraints
)

chairs, tables = solution.x

print("Optimization success:", solution.success)
print(f"Chairs to make: {chairs:.3f}")
print(f"Tables to make: {tables:.3f}")
print(f"Maximum profit: {-solution.fun:.2f}")
print(f"Wood used: {2 * chairs + 4 * tables:.2f} of {WOOD_AVAILABLE:.0f}")
print(f"Labour used: {3 * chairs + 2 * tables:.2f} of {LABOUR_AVAILABLE:.0f}")

best = None

for c in range(0, 41):
    for t in range(0, 26):
        if 2 * c + 4 * t <= WOOD_AVAILABLE and 3 * c + 2 * t <= LABOUR_AVAILABLE:
            profit = PROFIT_CHAIR * c + PROFIT_TABLE * t
            if best is None or profit > best[0]:
                best = (profit, c, t)

print("Best whole-number plan (profit, chairs, tables):", best)
