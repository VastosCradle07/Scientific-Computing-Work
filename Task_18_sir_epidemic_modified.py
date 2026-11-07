from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

N = 1000.0
beta = 0.4
gamma = 0.1


def sir(t, y):
    S, I, R = y
    return [
        -beta * S * I / N,
        beta * S * I / N - gamma * I,
        gamma * I
    ]


t_eval = np.linspace(0, 160, 641)

solution = solve_ivp(
    sir,
    (0, 160),
    [N - 1, 1.0, 0.0],
    t_eval=t_eval,
    rtol=1e-9,
    atol=1e-11
)

S, I, R = solution.y
peak_index = np.argmax(I)

print("Solver success:", solution.success)
print(f"Basic reproduction number R0: {beta / gamma:.2f}")
print(f"Peak infections: {I[peak_index]:.1f} people on day {solution.t[peak_index]:.1f}")
print(f"Total ever infected by day 160: {R[-1] + I[-1]:.1f} of {N:.0f}")
print(f"Still susceptible at day 160: {S[-1]:.1f}")
print(f"Largest deviation from population size: {np.max(np.abs(S + I + R - N)):.3e}")

plt.plot(solution.t, S, label="Susceptible")
plt.plot(solution.t, I, label="Infected")
plt.plot(solution.t, R, label="Recovered")
plt.xlabel("Day")
plt.ylabel("People")
plt.title("SIR epidemic model")
plt.legend()
plt.grid(True)
plt.savefig(OUTPUT_DIR / "Task_18_sir_epidemic_modified.png", dpi=150, bbox_inches="tight")
plt.show()
