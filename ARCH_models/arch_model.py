import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize


# -----------------------------------------
# 1) ARCH(1) Data Generator
# -----------------------------------------
def generate_arch1(n, phi_true):
    """
    Simulate ARCH(1) shocks and conditional variances.

    Parameters:
    - n: number of time steps
    - phi_true: true alpha1 (weight on past shock^2)

    Returns:
    - shocks: simulated shock series (epsilon_t)
    - cond_vars: true conditional variances (sigma_t^2)
    """
    alpha0 = 5.0  # baseline volatility parameter (alpha0)
    shocks = np.zeros(n)
    cond_vars = np.zeros(n)
    random_draws = np.random.normal(loc=0, scale=1, size=n + 1)

    # Initialize first period
    shocks[0] = 0.0
    cond_vars[0] = alpha0

    for t in range(1, n):
        cond_vars[t] = alpha0 + phi_true * (shocks[t - 1] ** 2)
        shocks[t] = random_draws[t] * math.sqrt(cond_vars[t])

    return shocks, cond_vars


# -----------------------------------------
# 2) Log-Likelihood for One Observation
# -----------------------------------------
def log_likelihood(shock, sig2):
    return -0.5 * (
        math.log(2 * math.pi) + math.log(sig2) + (shock ** 2) / sig2
    )


# -----------------------------------------
# 3) Loss function builder (Negative Log-Likelihood)
# -----------------------------------------
def make_loss(shocks):
    def loss(params):
        alpha0, alpha1 = params
        n = shocks.size
        cond_vars = np.zeros(n)
        cond_vars[0] = alpha0

        total_nll = 0.0
        for t in range(1, n):
            cond_vars[t] = alpha0 + alpha1 * (shocks[t - 1] ** 2)
            total_nll -= log_likelihood(shocks[t], cond_vars[t])

        return total_nll

    return loss


if __name__ == "__main__":
    # -----------------------------------------
    # 4) Simulate Data and Fit Parameters
    # -----------------------------------------
    N = 1000
    phi_true = 0.7

    # Generate shocks and true conditional variances
    shocks, true_vars = generate_arch1(N, phi_true)

    # Plot shocks (optional)
    plt.plot(shocks)
    plt.title("Simulated ARCH(1) Shocks")
    plt.xlabel("Time Step")
    plt.ylabel("Shock Value")
    plt.show()

    # Build loss function
    loss_fn = make_loss(shocks)

    # Bounds: alpha0 > 0, 0 <= alpha1 < 1
    bounds = [(1e-6, None), (0.0, 0.999)]
    initial_guess = [1.0, 0.5]

    # Run optimizer
    result = minimize(loss_fn, x0=initial_guess, bounds=bounds)

    print("Estimated alpha0 and alpha1:", result.x)
    print("Optimizer success:", result.success)

