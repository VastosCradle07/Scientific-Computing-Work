from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

P0 = 100.0
r = 0.4
K = 1000.0
end_time = 30.0


def growth_rate(P):
    return r * P * (1 - P / K)


def exact_solution(t):
    return K / (1 + (K - P0) / P0 * np.exp(-r * t))


def euler(h):
    time = np.arange(0, end_time + h / 2, h)
    population = np.zeros_like(time)
    population[0] = P0
    for i in range(len(time) - 1):
        population[i + 1] = population[i] + h * growth_rate(population[i])
    return time, population


plt.figure()

for h in (4.0, 1.0, 0.1):
    time, population = euler(h)
    error = abs(population[-1] - exact_solution(time[-1]))
    print(f"h={h}: population at t=30 is {population[-1]:.3f}, absolute error {error:.4f}")
    plt.plot(time, population, label=f"Euler h={h}")

fine_time = np.linspace(0, end_time, 400)
plt.plot(fine_time, exact_solution(fine_time), "k--", label="Exact")

print(f"Exact population at t=30: {exact_solution(end_time):.3f}")
print(f"Carrying capacity: {K:.0f}")

plt.xlabel("Time (years)")
plt.ylabel("Population")
plt.title("Logistic population growth with Euler steps")
plt.legend()
plt.grid(True)
plt.savefig(OUTPUT_DIR / "Task_17_logistic_euler_modified.png", dpi=150, bbox_inches="tight")
plt.show()
