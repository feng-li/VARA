import jax
import jax.numpy as jnp
from jax import grad, hessian
from jax.random import multivariate_normal, uniform, normal


# ---------- Attention builder ----------
def build_attention(Z, eta):
    p, q = Z.shape
    scores = (Z @ Z.T) / jnp.sqrt(q)
    scores = eta * scores
    mask = jnp.eye(p) * -1e9
    scores = scores + mask
    return jax.nn.softmax(scores, axis=1)


# ---------- Simulation function ----------
def simulate_VAR_attention(key, T=200, p=5, q=5, dist="normal"):
    keys = jax.random.split(key, 7)

    # Step 1: Feature vectors
    idx = jnp.arange(q)
    Sigma_z = 0.5 ** jnp.abs(idx[:, None] - idx[None, :])
    Z = multivariate_normal(keys[0], jnp.zeros(q), Sigma_z, (p,))

    # Step 2: Attention matrices
    W_a = build_attention(Z, eta=0.5)
    W_b = build_attention(Z, eta=1.0)

    # Step 3: Parameters
    gamma_a = uniform(keys[1], (p,), minval=-0.2, maxval=0.2)
    gamma_b = uniform(keys[2], (p,), minval=-0.2, maxval=0.2)
    beta    = uniform(keys[3], (p,), minval=-0.2, maxval=0.2)
    sigma2  = uniform(keys[4], (p,), minval=0.25, maxval=1.0)

    # Step 4: Noise
    eps_raw = normal(keys[5], (T, p))
    eps = eps_raw * jnp.sqrt(sigma2)

    # Step 5: Dynamics
    S = jnp.eye(p) - jnp.diag(gamma_b) @ W_b
    G = jnp.diag(beta) + jnp.diag(gamma_a) @ W_a
    S_inv = jnp.linalg.inv(S)

    Y = jnp.zeros((T, p))
    for t in range(1, T):
        Y = Y.at[t].set(S_inv @ (G @ Y[t-1] + eps[t]))

    # True parameter vector (ηa=0.5, ηb=1 used in sim)
    theta_true = jnp.concatenate([
        jnp.array([0.5, 1.0]),
        gamma_a, gamma_b, beta, sigma2
    ])

    return Y, theta_true, Z


# ---------- Log-likelihood with reparameterization ----------
def loglik_reparam(theta, Y, Z):
    """
    Reparameterized log-likelihood: last p entries are log(sigma2).
    """
    T, p = Y.shape
    eta_a = theta[0]
    eta_b = theta[1]
    gamma_a = theta[2:2+p]
    gamma_b = theta[2+p:2+2*p]
    beta    = theta[2+2*p:2+3*p]
    log_sigma2 = theta[2+3*p:2+4*p]
    sigma2 = jnp.exp(log_sigma2)   # enforce positivity

    W_a = build_attention(Z, eta_a)
    W_b = build_attention(Z, eta_b)

    D_beta = jnp.diag(beta)
    D_ga = jnp.diag(gamma_a)
    D_gb = jnp.diag(gamma_b)

    G = D_beta + D_ga @ W_a
    S = jnp.eye(p) - D_gb @ W_b

    inv_Sigma_e = jnp.diag(1.0 / sigma2)
    E = (Y[1:] @ S.T) - (Y[:-1] @ G.T)
    quad = jnp.sum((E @ inv_Sigma_e) * E)

    signS, logdetS = jnp.linalg.slogdet(S)
    if signS <= 0:
        return -jnp.inf  # invalid step

    val = (
        -(T-1) * p / 2 * jnp.log(2 * jnp.pi)
        - (T-1) / 2 * jnp.sum(log_sigma2)   # since sigma2 = exp(log_sigma2)
        + (T-1) * logdetS
        - 0.5 * quad
    )
    return val


# ---------- Newton-Raphson Estimator ----------
def mle_newton(Y, Z, theta_init, max_iter=50, tol=1e-6, alpha=1.0):
    loglik_grad = grad(loglik_reparam)
    loglik_hess = hessian(loglik_reparam)

    theta = theta_init
    for it in range(max_iter):
        g = loglik_grad(theta, Y, Z)
        H = loglik_hess(theta, Y, Z)

        try:
            step = jnp.linalg.solve(H, g)
        except jnp.linalg.LinAlgError:
            print("Hessian not invertible at iteration", it)
            break

        # line search / damping
        new_ll = loglik_reparam(theta + step, Y, Z)
        old_ll = loglik_reparam(theta, Y, Z)
        step_size = alpha
        while (not jnp.isfinite(new_ll)) or (new_ll < old_ll):
            step_size *= 0.5
            if step_size < 1e-6:
                print("Step size too small, stopping.")
                return theta, g, H
            new_ll = loglik_reparam(theta + step_size * step, Y, Z)

        theta_new = theta + step_size * step

        if jnp.linalg.norm(step_size * step) < tol:
            print(f"Converged in {it+1} iterations.")
            return theta_new, g, H

        theta = theta_new

    print("Max iterations reached.")
    return theta, g, H


# ---------- Example usage ----------
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    Y, theta_true, Z = simulate_VAR_attention(key, T=200, p=5, q=5, dist="normal")

    # Initial guess: perturb truth; log-transform sigma2 part
    p = Y.shape[1]
    log_sigma2_true = jnp.log(theta_true[2+3*p:2+4*p])
    theta_true_reparam = jnp.concatenate([
        theta_true[:2+3*p], log_sigma2_true
    ])
    theta_init = theta_true_reparam + 0.1 * jax.random.normal(key, theta_true_reparam.shape)

    theta_hat_reparam, grad_ll, hess_ll = mle_newton(Y, Z, theta_init)

    # Convert back to natural scale (replace log_sigma2 with sigma2)
    log_sigma2_hat = theta_hat_reparam[2+3*p:2+4*p]
    sigma2_hat = jnp.exp(log_sigma2_hat)
    theta_hat = jnp.concatenate([theta_hat_reparam[:2+3*p], sigma2_hat])

    print("True theta:", theta_true)
    print("Estimated theta:", theta_hat)
    print("Gradient norm at optimum:", jnp.linalg.norm(grad_ll))
    print("Hessian shape:", hess_ll.shape)
