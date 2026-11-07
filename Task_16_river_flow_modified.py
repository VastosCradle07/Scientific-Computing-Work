import numpy as np
from scipy.integrate import simpson

time_hours = np.arange(0, 13, dtype=float)
flow_m3s = np.array([12.0, 14.5, 19.0, 27.5, 38.0, 46.0, 49.5, 45.0, 36.0, 27.0, 20.0, 15.5, 13.0])

SECONDS_PER_HOUR = 3600.0

volume_trapezoid = np.trapezoid(flow_m3s, time_hours) * SECONDS_PER_HOUR
volume_simpson = simpson(flow_m3s, x=time_hours) * SECONDS_PER_HOUR

print(f"Volume using trapezoidal rule: {volume_trapezoid:,.0f} m3")
print(f"Volume using Simpson's rule: {volume_simpson:,.0f} m3")
print(f"Difference between estimates: {abs(volume_simpson - volume_trapezoid):,.0f} m3")
print(f"Average flow over 12 hours: {volume_simpson / (12 * SECONDS_PER_HOUR):.2f} m3/s")

x = np.linspace(0, np.pi, 9)
y = np.sin(x)
exact = 2.0

print()
print("Accuracy check on sin(x) from 0 to pi with 9 points (exact value 2)")
print(f"Trapezoidal error: {abs(np.trapezoid(y, x) - exact):.3e}")
print(f"Simpson error: {abs(simpson(y, x=x) - exact):.3e}")
