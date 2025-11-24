import numpy as np
import scipy
from tqdm import tqdm

# ----- Simulation settings -----
phi_true = 0.6     # true AR(1) coefficient
sigma_true = 1.0   # true innovation SD
T = 1000            # time series length
R = 10000          # number of replications
alpha = 0.1        # significance level

cover_count = 0    # to count coverage
interval_lengths = []

for _ in tqdm(range(R), desc="Simulating AR(1) coverage"):
    # ----- 1. Generate AR(1) data -----
    y = np.zeros(T)
    eps = np.random.normal(0, sigma_true, size=T)
    for t in range(1, T):
        y[t] = phi_true * y[t-1] + eps[t]

    # ----- 2. Estimate phi by OLS -----
    y_lag = y[:-1]
    y_curr = y[1:]
    phi_hat = np.sum(y_lag * y_curr) / np.sum(y_lag**2)

    # ----- 3. Estimate residual variance -----
    residuals = y_curr - phi_hat * y_lag
    sigma2_hat = np.sum(residuals**2) / (T - 2)

    # ----- 4. Standard error of phi_hat -----
    se_phi = np.sqrt(sigma2_hat / np.sum(y_lag**2))

    # ----- 5. Construct 95% confidence interval -----
    z = scipy.stats.norm.ppf(1 - alpha / 2.0)
    z = 1.64  # 95% level
    L = phi_hat - z * se_phi
    U = phi_hat + z * se_phi
    interval_lengths.append(U - L)

    # ----- 6. Check if true value is covered -----
    if (phi_true >= L) and (phi_true <= U):
        cover_count += 1

# ----- Compute empirical coverage -----
ecp = cover_count / R
avg_len = np.mean(interval_lengths)

print(f"Empirical coverage probability: {ecp:.3f}")
print(f"Average CI length: {avg_len:.3f}")
print(f"Nominal level: {1 - alpha:.2f}")
