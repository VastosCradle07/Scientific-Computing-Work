from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

rng = np.random.default_rng(42)

N = 200_000
deadline = 30.0

design = rng.triangular(5, 7, 12, N)
build = rng.triangular(10, 14, 25, N)
testing = rng.triangular(4, 6, 10, N)

total_days = design + build + testing

overrun = total_days > deadline

print(f"Mean project duration: {total_days.mean():.2f} days")
print(f"Standard deviation: {total_days.std():.2f} days")
print(f"90th percentile duration: {np.percentile(total_days, 90):.2f} days")
print(f"Probability of missing the {deadline:.0f}-day deadline: {overrun.mean():.4%}")
print(f"Number of overruns: {overrun.sum()} out of {N}")

M = 1_000_000
points = rng.random((M, 2))
inside = np.count_nonzero(np.sum(points ** 2, axis=1) <= 1.0)
pi_estimate = 4 * inside / M

print(f"Pi estimate with {M} points: {pi_estimate:.5f}")
print(f"Error against np.pi: {abs(pi_estimate - np.pi):.5f}")

plt.hist(total_days, bins=60, density=True, alpha=0.8)
plt.axvline(deadline, color="red", linestyle="--", label="Deadline")
plt.xlabel("Total duration (days)")
plt.ylabel("Probability density")
plt.title("Monte Carlo project duration")
plt.legend()
plt.grid(True)
plt.savefig(OUTPUT_DIR / "Task_20_project_risk_modified.png", dpi=150, bbox_inches="tight")
plt.show()
