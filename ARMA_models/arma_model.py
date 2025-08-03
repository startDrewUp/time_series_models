import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# Parameters
PHI = 0.5
THETA = 0.9
N = 10000
STARTING_VALUE = 0


def generate_arma1(phi, theta, n, starting_value):
    shocks = np.random.normal(0, 1, n + 1)
    values = np.zeros(n + 1)
    values[0] = starting_value

    for t in range(1, n + 1):
        values[t] = (
            phi * values[t - 1]
            + shocks[t]
            + theta * shocks[t - 1]
        )
    return shocks, values


def make_loss(values):
    def loss(params):
        phi, theta = params
        estimated_shocks = np.zeros_like(values)
        n = len(values)

        for t in range(1, n):
            predicted = phi * values[t - 1] + theta * estimated_shocks[t - 1]
            estimated_shocks[t] = values[t] - predicted

        mse = np.mean(estimated_shocks[1:] ** 2)  # skip t=0
        return mse

    return loss


if __name__ == "__main__":
    # Generate ARMA(1,1) data
    shocks, values = generate_arma1(PHI, THETA, N, STARTING_VALUE)

    # Plot the simulated series
    plt.plot(values)
    plt.title("Simulated ARMA(1,1) Time Series")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.show()

    # Estimate phi and theta with SciPy minimize
    loss_fn = make_loss(values)
    initial_guess = [0.1, 0.1]
    result = minimize(loss_fn, initial_guess)

    print("Estimated φ:", result.x[0])
    print("Estimated θ:", result.x[1])
