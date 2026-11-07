from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

time = np.arange(0, 11, dtype=float)
velocity = np.array([0.0, 2.1, 4.3, 6.8, 9.0, 11.2, 13.1, 14.8, 16.0, 16.8, 17.0])

acceleration = np.gradient(velocity, time)

for t, v, a in zip(time, velocity, acceleration):
    print(f"t={t:.0f}s, velocity={v:.1f}m/s, estimated acceleration={a:.2f}m/s2")

print(f"Peak acceleration: {acceleration.max():.2f} m/s2 at t={time[np.argmax(acceleration)]:.0f}s")

x0 = 1.0
exact = np.cos(x0)

print()
print("Step-size study for the derivative of sin(x) at x=1")
print(f"{'h':>8} {'forward error':>16} {'central error':>16}")
for h in (0.1, 0.01, 0.001):
    forward = (np.sin(x0 + h) - np.sin(x0)) / h
    central = (np.sin(x0 + h) - np.sin(x0 - h)) / (2 * h)
    print(f"{h:>8} {abs(forward - exact):>16.3e} {abs(central - exact):>16.3e}")

plt.plot(time, acceleration, marker="o")
plt.axhline(0, linewidth=0.8)
plt.xlabel("Time (s)")
plt.ylabel("Estimated acceleration (m/s2)")
plt.title("Acceleration estimated from velocity data")
plt.grid(True)
plt.savefig(OUTPUT_DIR / "Task_15_acceleration_modified.png", dpi=150, bbox_inches="tight")
plt.show()
