from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

months = np.arange(1, 11, dtype=float)
users_thousands = np.array([12, 15, 19, 24, 30, 37, 47, 58, 73, 91], dtype=float)


def exponential_model(x, a, b):
    return a * np.exp(b * x)


def linear_model(x, m, c):
    return m * x + c


exp_params, _ = curve_fit(exponential_model, months, users_thousands, p0=(10.0, 0.2))
lin_params, _ = curve_fit(linear_model, months, users_thousands)

exp_pred = exponential_model(months, *exp_params)
lin_pred = linear_model(months, *lin_params)


def rmse(observed, predicted):
    return np.sqrt(np.mean((observed - predicted) ** 2))


def r_squared(observed, predicted):
    residual_sum = np.sum((observed - predicted) ** 2)
    total_sum = np.sum((observed - observed.mean()) ** 2)
    return 1 - residual_sum / total_sum


a, b = exp_params

print(f"Exponential model: users = {a:.3f} * exp({b:.4f} * month)")
print(f"Monthly growth rate: {np.exp(b) - 1:.2%}")
print(f"Doubling time: {np.log(2) / b:.2f} months")
print(f"Exponential RMSE: {rmse(users_thousands, exp_pred):.2f} thousand, R2 = {r_squared(users_thousands, exp_pred):.4f}")
print(f"Linear RMSE: {rmse(users_thousands, lin_pred):.2f} thousand, R2 = {r_squared(users_thousands, lin_pred):.4f}")
print(f"Predicted users in month 12 (exponential): {exponential_model(12, *exp_params):.1f} thousand")
print(f"Predicted users in month 12 (linear): {linear_model(12, *lin_params):.1f} thousand")

x_plot = np.linspace(1, 12, 300)

plt.scatter(months, users_thousands, label="Observed users")
plt.plot(x_plot, exponential_model(x_plot, *exp_params), label="Exponential fit")
plt.plot(x_plot, linear_model(x_plot, *lin_params), linestyle="--", label="Linear fit")
plt.xlabel("Month")
plt.ylabel("Users (thousands)")
plt.title("Mobile app user growth")
plt.legend()
plt.grid(True)
plt.savefig(OUTPUT_DIR / "Task_14_app_growth_modified.png", dpi=150, bbox_inches="tight")
plt.show()
