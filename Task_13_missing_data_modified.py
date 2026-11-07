from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline, interp1d

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def true_temperature(t):
    return 22 + 6 * np.sin(np.pi * (t - 6) / 12)


hours = np.array([0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12], dtype=float)
temperature = true_temperature(hours)

missing = np.array([5.0, 6.0])

linear = interp1d(hours, temperature, kind="linear")
cubic = CubicSpline(hours, temperature)

truth = true_temperature(missing)

print("Missing hours:", missing)
print("True values:", np.round(truth, 4))
print("Linear estimates:", np.round(linear(missing), 4))
print("Cubic estimates:", np.round(cubic(missing), 4))
print("Linear absolute errors:", np.round(np.abs(linear(missing) - truth), 4))
print("Cubic absolute errors:", np.round(np.abs(cubic(missing) - truth), 4))

fine = np.linspace(0, 12, 241)

plt.scatter(hours, temperature, label="Recorded")
plt.scatter(missing, truth, marker="x", color="red", label="Missing (true)")
plt.plot(fine, linear(fine), label="Linear")
plt.plot(fine, cubic(fine), label="Cubic spline")
plt.xlabel("Hour of day")
plt.ylabel("Temperature (C)")
plt.title("Filling a gap in sensor data")
plt.legend()
plt.grid(True)
plt.savefig(OUTPUT_DIR / "Task_13_missing_data_modified.png", dpi=150, bbox_inches="tight")
plt.show()
