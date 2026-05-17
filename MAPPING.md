# Parameter Mapping Between Paper and Code

This repository still contains some historical code names. The simulation
tables and manuscript use paper notation, while lower-level DGP and estimation
code use the original implementation notation.

## Core Model

Paper model:

```text
y_t = alpha * y_{t-1} + beta * W(gamma; Z) y_{t-1} + epsilon_t
```

Code model:

```text
Y_t = beta * Y_{t-1} + gamma_a * W(eta_a; Z) Y_{t-1} + eps_t
```

## Name Mapping

| Paper symbol | Meaning | Code name | Shape |
|---|---|---|---|
| `gamma` | attention sharpness / feature-distance parameter | `eta_a` | scalar |
| `alpha` | own-lag coefficient | `beta` | `(p,)` |
| `beta` | attention/network-lag coefficient | `gamma_a` | `(p,)` |
| `sigma^2` | innovation variance | `sigma2` | `(p,)` |

## Estimator Output Mapping

The two-stage estimator returns paper-notation names:

| Estimator output | Estimates paper symbol | Estimates code parameter |
|---|---|---|
| `eta_hat` | `gamma` | `eta_a` |
| `alpha_hat` | `alpha` | `beta` |
| `beta_hat` | `beta` | `gamma_a` |
| `sigma2_hat` | `sigma^2` | `sigma2` |

## Simulation Table Mapping

The simulation summaries and manuscript tables use paper notation:

| Table column | Source row metric | Paper symbol | Code parameter |
|---|---|---|---|
| `RMSE_gamma` | `eta_sqerr` | `gamma` | `eta_a` |
| `RMSE_alpha` | `alpha_mse` | `alpha` | `beta` |
| `RMSE_beta` | `beta_mse` | `beta` | `gamma_a` |
| `RMSE_sigma2` | `sigma2_mse` | `sigma^2` | `sigma2` |
| `ECP_gamma` | `eta_covered` | `gamma` | `eta_a` |
| `ECP_alpha` | `alpha_ecp` | `alpha` | `beta` |
| `ECP_beta` | `beta_ecp` | `beta` | `gamma_a` |
| `ECP_sigma2` | `sigma2_ecp` | `sigma^2` | `sigma2` |

## DGP Ranges in `scripts/run_full_simulation.py`

Current full-simulation DGP ranges are:

| Paper symbol | Code parameter | Current draw |
|---|---|---|
| `gamma` | `eta_a` | fixed by `--dgp-eta-a`, default manuscript value |
| `alpha` | `beta` | uniform on `[-0.20, 0.20]` |
| `beta` | `gamma_a` | uniform on `[0.35, 0.45]` |
| `sigma^2` | `sigma2` | uniform on `[0.50, 1.00]` |

The stationarity check rescales code `beta` and `gamma_a` together only if
needed.

## Practical Reading Guide

When paper `beta` has a large RMSE, inspect code `gamma_a` / estimator
`beta_hat`. This is the attention-weighted lag coefficient, not the own-lag
coefficient.

When paper `alpha` has a small RMSE, inspect code `beta` / estimator
`alpha_hat`. This is the own-lag coefficient.

