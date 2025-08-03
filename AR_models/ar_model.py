import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# Parameters
PHI = 0.7
N = 1000
STARTING_VALUE = 0


def generate_ar1(phi, n, starting_value):
    values = np.zeros(n + 1)  # length n+1
    shocks = np.random.normal(0, 1, n)  # n shocks

    values[0] = starting_value
    for t in range(1, n + 1):
        values[t] = phi * values[t - 1] + shocks[t - 1]

    return shocks, values


def estimate_phi(values):
    numerator = np.dot(values[1:], values[:-1])
    denominator = values[:-1] ** 2
    sum_denominator = np.sum(denominator)
    return numerator / sum_denominator


def make_loss(values):
    def loss(phi):
        mse = 0
        for t in range(1, values.size):
            mse += (values[t] - phi * values[t - 1]) ** 2
        return mse / (values.size - 1)

    return loss


if __name__ == "__main__":
    # Generate data
    shocks, values = generate_ar1(PHI, N, STARTING_VALUE)

    # Plot time series
    plt.plot(values)
    plt.title("Simulated AR(1) Time Series")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.show()

    # Estimate phi (OLS)
    estimated_phi_ols = estimate_phi(values)
    print(f"Estimated φ (OLS): {estimated_phi_ols:.4f}")

    # Estimate phi (SciPy minimize)
    phi0 = [0.5]
    loss_fn = make_loss(values)
    result = minimize(loss_fn, phi0)
    print(f"Estimated φ (SciPy minimize): {result.x[0]:.4f}")

