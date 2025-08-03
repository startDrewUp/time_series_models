import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.optimize import minimize
import random
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib
matplotlib.use("TkAgg")


PHI = 0.6
N = 1000


def generate_ma1(phi, n):
    ma1_values = np.zeros(n)
    ma1_shocks = np.random.normal(0, 1, n)
    for t in range(n):
        ma1_values[t] = ma1_shocks[t] + phi * ma1_shocks[t - 1]
    return ma1_values, ma1_shocks


def make_loss(values):
    ERROR0 = 0
    def loss(phi):
        estimated_shocks = np.zeros(len(values))
        estimated_shocks[0] = ERROR0
        for t in range(1, len(values)):
            estimated_shocks[t] = values[t] - (phi[0] * estimated_shocks[t - 1])
        return np.sum(estimated_shocks ** 2)
    return loss


if __name__ == "__main__":
    # Generate data
    values, shocks = generate_ma1(PHI, N)

    # Plot time series
    plt.plot(values)
    plt.title("Simulated MA(1) Time Series")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    ## plt.show()

    # Plot ACF
    plot_acf(values, lags=20)
    plt.title("ACF of Simulated MA(1) Data")
    ## plt.show()

    # Estimate phi
    phi0 = [0.5]  # initial guess
    loss_fn = make_loss(values)
    result = minimize(loss_fn, phi0)
    print("Estimated θ:", result.x[0])

