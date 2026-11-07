import numpy as np

tank_capacity_l = 5000.0

levels_l = np.array([
    4200, 4100, 4000, 3900, 3850, 3800,
    3500, 3000, 2600, 2400, 2300, 2500,
    2800, 3200, 3600, 3900, 4100, 4000,
    3700, 3300, 3000, 2900, 3200, 3500
], dtype=float)

mean_level = np.mean(levels_l)
min_level = np.min(levels_l)
max_level = np.max(levels_l)

percent_full = levels_l / tank_capacity_l * 100
volume_m3 = levels_l / 1000
hourly_change = np.diff(levels_l)
low_hours = np.count_nonzero(percent_full < 50)

print(f"Mean level: {mean_level:.1f} L ({mean_level / 1000:.3f} m3)")
print(f"Minimum level: {min_level:.0f} L at hour {np.argmin(levels_l)}")
print(f"Maximum level: {max_level:.0f} L at hour {np.argmax(levels_l)}")
print("First six readings as percent full:", np.round(percent_full[:6], 1))
print("First six readings in m3:", volume_m3[:6])
print(f"Largest hourly drop: {hourly_change.min():.0f} L")
print(f"Largest hourly rise: {hourly_change.max():.0f} L")
print(f"Hours below 50 percent full: {low_hours} of {levels_l.size}")
