# Vector Autoregression with Attention Weights (VARA)

This repository implements a **network-augmented Vector Autoregression (VAR)** model that incorporates structured dependencies through attention-like weight matrices.

## Model

The model is defined as:

$$
Y_t = \text{diag}(\beta) Y_{t-1}
+ \text{diag}(\gamma_a) W_a Y_{t-1}
+ \text{diag}(\gamma_b) W_b Y_t
+ \varepsilon_t,
$$

where

* $Y_t \in \mathbb{R}^p$: vector of (p) endogenous time series at time (t).
* $\text{diag}(\beta)$: diagonal autoregressive coefficient matrix for own-lag effects.
* $W_a$: attention/weight matrix capturing **cross-series lagged dependence**.
* $\text{diag}(\gamma_a)$: scaling coefficients for the weighted lag term.
* $W_b): attention/weight matrix capturing **contemporaneous dependence**.
* $\text{diag}(\gamma_b)$: scaling coefficients for contemporaneous effects.
* $\varepsilon_t$: innovation vector.

## Key Features

* **Own-lag dynamics**: Each series retains its traditional VAR autoregressive structure.
* **Attention-weighted lag effects**: Cross-series interactions at lag 1 are governed by (W_a), which can be learned or externally specified (e.g., economic networks, spatial adjacency, learned attention scores).
* **Contemporaneous interactions**: (W_b) allows modeling of instantaneous effects across series, beyond traditional VAR specifications.
* **Scalable**: By using structured matrices, the model reduces the parameter explosion of unrestricted VARs, making it suitable for high-dimensional time series.

## Applications

* **Macroeconomic forecasting** (capturing spillovers between sectors or countries).
* **Financial networks** (modeling interdependencies of assets, systemic risk).
* **Spatio-temporal forecasting** (regional demand, transportation, energy consumption).
* **Sensor networks** (IoT, environmental monitoring).
