import timeit

import numpy as np

rng = np.random.default_rng(7)
prices_local = rng.uniform(100, 50_000, 2_000_000)

EXCHANGE_RATE = 0.0077
VAT_RATE = 0.16


def loop_version():
    totals = []
    for price in prices_local:
        usd = price * EXCHANGE_RATE
        totals.append(usd * (1 + VAT_RATE))
    return totals


def vectorized_version():
    return prices_local * EXCHANGE_RATE * (1 + VAT_RATE)


loop_time = timeit.timeit(loop_version, number=1)
vector_time = timeit.timeit(vectorized_version, number=3) / 3

result = vectorized_version()
loop_result = np.array(loop_version())

print("First five totals in USD:", np.round(result[:5], 2))
print(f"Grand total in USD: {result.sum():,.2f}")
print("Both versions agree:", np.allclose(result, loop_result))
print(f"Loop time: {loop_time:.4f} s")
print(f"Vectorized time: {vector_time:.4f} s")
print(f"Approximate speed-up: {loop_time / vector_time:.1f}x")
