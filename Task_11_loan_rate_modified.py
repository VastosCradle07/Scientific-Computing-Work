from scipy.optimize import root_scalar

LOAN = 10000.0
PAYMENT = 920.0
MONTHS = 12


def loan_gap(rate):
    return PAYMENT * (1 - (1 + rate) ** -MONTHS) / rate - LOAN


def d_loan_gap(rate, step=1e-7):
    return (loan_gap(rate + step) - loan_gap(rate - step)) / (2 * step)


newton = root_scalar(
    loan_gap,
    fprime=d_loan_gap,
    x0=0.01,
    method="newton",
    xtol=1e-12
)

secant = root_scalar(
    loan_gap,
    x0=0.005,
    x1=0.03,
    method="secant",
    xtol=1e-12
)

monthly = newton.root
effective_annual = (1 + monthly) ** 12 - 1

print(f"Monthly rate using Newton-Raphson: {monthly:.6%}")
print(f"Monthly rate using Secant method: {secant.root:.6%}")
print(f"Newton iterations: {newton.iterations}")
print(f"Secant iterations: {secant.iterations}")
print(f"Nominal annual rate: {12 * monthly:.4%}")
print(f"Effective annual rate: {effective_annual:.4%}")
print(f"Total interest paid: {PAYMENT * MONTHS - LOAN:.2f}")
print(f"Gap at Newton solution: {loan_gap(monthly):.2e}")
