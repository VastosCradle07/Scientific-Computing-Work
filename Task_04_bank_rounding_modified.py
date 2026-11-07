import math
from decimal import Decimal

import numpy as np

payments = 1_000_000
deposit = 0.1
expected_total = 100_000.0

naive_total = 0.0
for _ in range(payments):
    naive_total += deposit

fsum_total = math.fsum([deposit] * payments)
numpy_total = np.sum(np.full(payments, deposit))
decimal_total = Decimal("0.1") * payments

print(f"Naive loop total: {naive_total:.10f}")
print(f"math.fsum total: {fsum_total:.10f}")
print(f"NumPy pairwise total: {numpy_total:.10f}")
print(f"Decimal total: {decimal_total}")

for label, value in (("naive", naive_total), ("fsum", fsum_total), ("numpy", numpy_total)):
    absolute_error = abs(value - expected_total)
    relative_error = absolute_error / expected_total
    print(f"{label}: absolute error = {absolute_error:.3e}, relative error = {relative_error:.3e}")

print("Naive total equals expected exactly:", naive_total == expected_total)
print("Naive total close to expected:", np.isclose(naive_total, expected_total))

balance = 1000.0
for _ in range(12):
    balance *= 1 + 0.015
print(f"Balance after 12 months at 1.5 percent: {balance:.6f}")
print(f"Rounded to cents: {round(balance, 2):.2f}")
