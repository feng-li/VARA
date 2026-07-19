# Instructions for Applying VARA to an OptionMetrics IvyDB US Implied-Volatility Panel

## 1. Purpose

This document specifies a reproducible financial-data application of **Vector Autoregression with Attention (VARA)** using the standardized implied-volatility surfaces in **OptionMetrics IvyDB US**. The design is intended to be close to the scale of the Walmart M5 application in the VARA manuscript while giving the attention features a direct financial interpretation.

The default panel has:

| Quantity | Design |
|---|---:|
| Underlyings | 500 liquid U.S. optionable securities |
| Surface nodes per underlying | 60 |
| Total series, `p` | 30,000 |
| Daily observations, `T` | 1,941 trading days |
| Total panel cells | 58,230,000 |
| M5 comparison | 30,490 series × 1,941 days |
| Primary response | Standardized residual of log implied volatility |
| Default forecast task | One-day-ahead rolling forecasts |
| M5-comparable task | A fixed 28-trading-day validation block followed by a fixed 28-trading-day test block |

The VARA model, attention formula, likelihood, and two-stage estimator are not changed. All changes occur in data selection, preprocessing, feature construction, baseline forecasting, validation, and forecast combination.

---

## 2. Data source and access

### 2.1 Primary source

Use **OptionMetrics IvyDB US**, a licensed database of end-of-day U.S. equity, ETF, ADR, and index option data. The database includes implied volatilities, option sensitivities, underlying prices, option volume and open interest, permanent security identifiers, and standardized constant-maturity volatility surfaces.

Official source:

- OptionMetrics IvyDB US: <https://optionmetrics.com/united-states/>

Institutional access is often provided through **Wharton Research Data Services (WRDS)**:

- WRDS OptionMetrics page: <https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/optionmetrics/>

The product is proprietary. Follow the institution's subscription and redistribution rules. Share extraction code, variable definitions, and aggregate results, but do not redistribute licensed raw records unless the license explicitly permits it.

### 2.2 Documentation

Use the current IvyDB US data dictionary supplied with the subscription as the authoritative schema. An archived IvyDB US Reference Manual v5.0 is publicly viewable here:

- Archived reference manual: <https://lib.tsinghua.edu.cn/PDF/OptionMetricsshiyongzhinan.pdf>

Field names and table names can differ between an OptionMetrics flat-file delivery, WRDS, and different IvyDB releases. The remainder of this document therefore uses **canonical names**. Create a schema-mapping file that records the exact source table and source column corresponding to each canonical variable.

### 2.3 Data snapshot

Record all of the following before extraction:

```yaml
vendor: OptionMetrics
product: IvyDB US
access_platform: WRDS_or_direct_delivery
product_release: <release number>
extraction_date: YYYY-MM-DD
data_revision_cutoff: YYYY-MM-DD
source_schema: <WRDS schema or flat-file release>
```

OptionMetrics can issue corrections to historical data. A frozen extraction date is needed for reproducibility.

---

## 3. Required source files or tables

The exact names depend on the delivery platform. The canonical source objects are:

| Canonical source object | Typical IvyDB flat-file name | Required? | Main use |
|---|---|---:|---|
| `volatility_surface` | `IVYVSURF` / Volatility_Surface | Yes | Daily standardized implied-volatility nodes |
| `security_master` | `IVYSECUR` / Security | Yes | Permanent ID, issue type, industry information |
| `security_name_history` | `IVYSECNM` / Security_Name | Yes | Historical ticker, issuer, CUSIP, SIC |
| `security_price` | `IVYSECPR` / Security_Price | Yes | Underlying price, return, volume, shares outstanding |
| `option_volume` | `IVYOPVOL` / Option_Volume | Recommended | Call/put option volume and open interest |
| `historical_volatility` | `IVYHVOL` / Historical_Volatility | Recommended | Vendor historical-volatility measures |
| `exchange_history` | `IVYEXCHG` / Exchange | Optional | Listing and exchange-history features |
| `option_price` | `IVYOPPRC` / Option_Price | Robustness only | Quote-derived surface checks using actual contracts |

Do not use the current ticker as the panel key. Use the permanent OptionMetrics security identifier, `secid`, because tickers and names can change.

---

## 4. Canonical raw-variable dictionary

### 4.1 Volatility-surface variables

| Canonical variable | Source meaning | Required cleaning and interpretation |
|---|---|---|
| `secid` | Permanent identifier of the underlying security | Store as an integer or string key; never treat its numeric value as a feature. |
| `date` | End-of-day observation date | Convert to a trading-date type and sort strictly within `secid`. |
| `days` | Calendar days to the standardized expiration | The selected grid is defined in Section 5. |
| `delta_raw` | Delta grid value in the vendor representation | Some extracts use integers such as `20`; others use decimals such as `0.20`. Puts can be stored with negative delta. Normalize explicitly. |
| `cp_flag` | Option type: call or put | Canonical values must be `C` and `P`. |
| `impl_volatility_raw` | Interpolated implied volatility at the standardized surface node | Convert missing sentinels and nonpositive values to missing. Verify whether the local extract is in decimal or percentage units. Store canonical IV in annualized decimal units. |
| `impl_strike` | Strike corresponding to the standardized delta node | Use for quality checks and optional moneyness diagnostics; it is not the forecasting target. |
| `impl_premium` | Theoretical premium for the standardized option | Optional feature and quality diagnostic. |
| `dispersion` | Vendor measure of uncertainty or dispersion in the smoothed surface estimate | Larger values indicate a less precise surface estimate. Values such as `-99.99` are missing codes in documented releases. |

The standardized volatility surface is produced only when the vendor has enough option information to interpolate the node. Therefore, missingness is economically informative and must be measured rather than silently discarded.

### 4.2 Security-master variables

| Canonical variable | Source meaning | Use |
|---|---|---|
| `ticker` | Ticker valid for a particular effective date | Labels and reporting only. |
| `cusip8` | Eight-character CUSIP valid for a particular effective date | Cross-database linkage, subject to licensing. |
| `issuer_description` | Issuer name | Labels and manual validation. |
| `issue_description` | Security description | Asset-type validation. |
| `sic` | Standard Industrial Classification code | Construct coarse sector and industry features. |
| `industry_group` | Vendor industry-group code | Construct sector/business-segment indicators. |
| `index_flag` | Indicator that the underlying is an index | Universe filter or component feature. |
| `issue_type` | Security type, such as common equity, ADR, ETF, or fund | Universe filter and one-hot feature. |
| `exchange` | Active listing exchange or index designation | Optional one-hot feature. |
| `listing_status` | Listing, delisting, suspension, acquisition, or other status | Prevent accidental use of observations after the instrument becomes inactive. |

Historical name records must be joined **as of each date**. Do not attach a future ticker, SIC code, or description to an earlier observation.

### 4.3 Underlying-price variables

| Canonical variable | Source meaning | Required cleaning and interpretation |
|---|---|---|
| `close_price_raw` | Closing price or, in documented cases, a signed convention indicating a bid-ask midpoint | Create `close_price = abs(close_price_raw)` after verifying the local documentation. Preserve a flag indicating whether the observation was a traded close or midpoint. |
| `open_price` | Opening price when available | Optional feature. |
| `bid_low` | Daily low, or a signed bid convention when no trade occurred | Use only after consulting the release-specific documentation. |
| `ask_high` | Daily high, or a signed ask convention when no trade occurred | Use only after consulting the release-specific documentation. |
| `underlying_volume` | Aggregate share volume in the underlying | Apply `log1p` for feature construction. |
| `total_return` | Vendor holding-period total return from the last good pricing date | Use as the preferred return measure after checking for stale-price intervals. |
| `shares_outstanding` | Public shares outstanding, generally lagged in the vendor file | Construct market capitalization. |
| `adjustment_factor` | Cumulative split/distribution adjustment factor | Retain for data validation and optional reconstruction. |

Define

\[
\text{market\_cap}_{t,u}=\text{close\_price}_{t,u}\times\text{shares\_outstanding}_{t,u}.
\]

### 4.4 Option-market liquidity variables

| Canonical variable | Source meaning | Use |
|---|---|---|
| `option_volume_call` | Total daily contract volume in calls for an underlying | Liquidity and recent-activity features. |
| `option_volume_put` | Total daily contract volume in puts for an underlying | Liquidity and put-call activity features. |
| `option_volume_total` | Total call-plus-put contract volume | Universe ranking and liquidity features. |
| `open_interest_call` | Aggregate call open interest | Liquidity feature. |
| `open_interest_put` | Aggregate put open interest | Liquidity feature. |
| `open_interest_total` | Aggregate call-plus-put open interest | Universe ranking and liquidity feature. |

### 4.5 Historical-volatility variables

| Canonical variable | Source meaning | Use |
|---|---|---|
| `hv_days` | Lookback length in calendar days | Select documented horizons such as 30, 60, 91, 182, or 365 days. |
| `historical_volatility` | Annualized realized volatility computed by the vendor | Underlying-level feature and IV-minus-HV spread diagnostic. |

---

## 5. Panel construction

### 5.1 Surface grid

Use the following default grid because it gives exactly 60 nodes per underlying while avoiding the least liquid extreme tails and the longest two-year maturity:

```text
maturities_days = [30, 60, 91, 122, 182, 365]
absolute_deltas = [0.20, 0.35, 0.50, 0.65, 0.80]
option_types     = [C, P]
```

The number of nodes is

\[
6\times5\times2=60.
\]

Normalize delta as follows:

```python
if abs(delta_raw) > 1:
    delta_decimal = delta_raw / 100.0
else:
    delta_decimal = delta_raw

absolute_delta = abs(delta_decimal)
signed_delta = absolute_delta if cp_flag == "C" else -absolute_delta
```

Reject a record if the option type and delta sign are logically inconsistent after consulting the release-specific codebook.

### 5.2 Sample length and split

Construct exactly 1,941 ordered trading dates from the dates present in the data. Do not create synthetic weekend or holiday rows.

For the M5-comparable fixed-origin experiment:

| Block | Number of dates | Use |
|---|---:|---|
| Inner training | 1,885 | Universe selection, feature construction, baseline and VARA tuning inputs |
| Validation | 28 | Select feature specification, ridge penalty, attention settings, and residual shrinkage |
| Final test | 28 | Untouched final evaluation |
| Total | 1,941 | Complete application panel |

The final 28 dates are the test set. The preceding 28 dates are validation. All earlier dates are inner training.

For the preferred financial forecasting experiment, retain a longer sequence of one-day-ahead forecast origins, ideally 126 or 252 trading days, because VARA is a lag-one model and financial dependence is strongest at short horizons. Keep the last 28-day experiment as a direct scalability comparison with M5.

### 5.3 Underlying universe

Use only information from the inner-training block to choose the 500 underlyings.

Default eligible universe:

1. U.S. common equities with listed options.
2. Valid permanent `secid` throughout the relevant training observations.
3. Valid underlying prices and returns.
4. Sufficient standardized-surface coverage on the selected 60-node grid.

A mixed equity/ETF/index universe can be reported as an ablation, but the primary panel should be homogeneous unless there is a strong substantive reason to mix asset types.

For each candidate underlying `u`, compute on the inner-training block:

\[
\text{coverage}_u=
\frac{\#\{(t,n):\text{node }n\text{ has valid IV for }u\}}
{1885\times60},
\]

\[
L_u=\operatorname{median}_t\log\{1+\text{option volume}_{t,u}\}
+\operatorname{median}_t\log\{1+\text{open interest}_{t,u}\}.
\]

Recommended filters:

```text
overall surface coverage >= 0.98
coverage for every selected node >= 0.95
valid underlying return fraction >= 0.98
active on the last inner-training date
```

Rank the eligible underlyings by a preregistered liquidity score, for example the average of the percentile ranks of coverage, median log option volume, and median log open interest. Select the top 500 and freeze the universe before validation.

Do not use a current S&P 500 constituent list. Doing so creates survivorship bias.

### 5.4 Component identifier

Each VARA component is one fixed underlying-surface node:

```text
component_id = secid | cp_flag | days | absolute_delta
```

Example:

```text
101594|P|91|0.20
```

Create a permanent integer `component_index` from 0 to 29,999. Save a component map with:

```text
component_index
component_id
secid
cp_flag
days
absolute_delta
signed_delta
ticker_at_training_end
issuer_description
sector_code
issue_type
```

The modeling matrix must have shape `(T, p)` with dates in rows and `component_index` in columns.

### 5.5 Missing values and quality control

VARA requires a balanced numeric panel. Use the following leakage-safe procedure.

1. Convert documented missing sentinels, nonfinite values, and nonpositive IV values to missing.
2. Verify uniqueness of `(secid, date, days, absolute_delta, cp_flag)`.
3. For a missing surface node on date `t`, first interpolate within the same underlying, option type, and date using neighboring delta and log-maturity nodes. Use only observations available on date `t`.
4. If interpolation is impossible, carry the previous valid value forward for at most two trading days.
5. Never backfill from a future date.
6. If an underlying still has any long missing run in inner training, remove it and replace it with the next eligible underlying in the preregistered ranking.
7. Save `was_interpolated` and `was_forward_filled` flags. Use their training fractions as component features.
8. In validation and test, preserve the fixed component set. If the realized target remains missing after the allowed procedure, exclude that cell from the accuracy metric and report the matched-cell count. Do not replace an unobserved target merely to increase coverage.

Additional quality checks:

```text
0 < implied volatility
0 < close price
market capitalization > 0 when used
no duplicate component-date records
strictly increasing dates
all model-input cells finite after preprocessing
```

Winsorize extreme values using cutoffs estimated from inner training only. A reasonable default is the 0.1st and 99.9th percentiles of `log_iv` changes, applied componentwise with a minimum sample-size safeguard.

---

## 6. Modeling variables

### 6.1 Implied-volatility target

Let `iv[t,k]` be annualized implied volatility in decimal units. Define

\[
y_{t,k}=\log(\text{iv}_{t,k}).
\]

The logarithm enforces positive inverse-transformed forecasts and reduces scale differences between low- and high-volatility securities.

### 6.2 Baseline forecast and residual

Fit a strong univariate baseline separately to every component. Let

\[
\widehat y^{\text{base}}_{t,k}
\]

be the baseline fitted or forecast value, and define the residual or genuine historical forecast error

\[
r_{t,k}=y_{t,k}-\widehat y^{\text{base}}_{t,k}.
\]

Genuine rolling one-step-ahead errors are preferred to in-sample residuals. Every error dated `t` must be generated by a baseline fitted only through `t-1`.

### 6.3 Robust residual standardization

For each component, estimate the residual center and scale from the current training window:

\[
m_k=\operatorname{median}_t(r_{t,k}),
\qquad
s_k=1.4826\operatorname{median}_t|r_{t,k}-m_k|.
\]

Use a scale floor:

\[
s_k\leftarrow\max(s_k,0.05\times\operatorname{median}_{j}s_j,10^{-6}).
\]

The VARA response is

\[
z_{t,k}=\frac{r_{t,k}-m_k}{s_k}.
\]

Save `m_k` and `s_k`. They are needed to return the VARA residual forecast to the log-IV scale.

### 6.4 Canonical processed-variable dictionary

| Variable | Definition |
|---|---|
| `iv` | Clean annualized implied volatility in decimal units. |
| `log_iv` | `log(iv)`. |
| `dlog_iv` | First difference of `log_iv`. |
| `baseline_hat` | Baseline fitted value or forecast on the log-IV scale. |
| `baseline_error` | `log_iv - baseline_hat`. |
| `resid_center` | Training-only median of `baseline_error` for the component. |
| `resid_scale` | Training-only robust scale of `baseline_error`. |
| `vara_response` | Standardized residual `(baseline_error-resid_center)/resid_scale`. |
| `vara_forecast_std` | VARA forecast of the standardized residual. |
| `vara_correction` | `resid_center + resid_scale * vara_forecast_std`. |
| `lambda` | Validation-selected residual-correction shrinkage. |
| `final_log_iv_hat` | `baseline_hat + lambda * vara_correction`. |
| `final_iv_hat` | `exp(final_log_iv_hat)`. |

---

## 7. Baseline models

### 7.1 Primary baseline: componentwise HAR model

For each component, estimate

\[
y_{t,k}=c_k+
\phi_{1,k}y_{t-1,k}+
\phi_{5,k}\overline y^{(5)}_{t-1,k}+
\phi_{22,k}\overline y^{(22)}_{t-1,k}+u_{t,k},
\]

where

\[
\overline y^{(m)}_{t-1,k}=m^{-1}\sum_{j=1}^{m}y_{t-j,k}.
\]

Use OLS or a small ridge penalty selected in training. The HAR baseline is efficient to refit and captures daily, weekly, and monthly persistence.

### 7.2 Required benchmarks

Report at least:

1. Random walk in log IV: `forecast = last log_iv`.
2. AR(1) per component.
3. HAR per component.
4. AutoETS per component, for comparability with the M5 application.
5. HAR plus univariate AR(1) residual correction.
6. VARA on HAR residuals.
7. VARA on AutoETS residuals as a robustness specification.

Use exactly the same information set and evaluation cells for all methods.

---

## 8. Component-feature construction for attention

VARA uses a static component vector `Z[k]` within each fitted forecasting origin. “Static” means fixed during that fit, not fixed forever. Recompute all training-derived features when the forecasting origin changes.

### 8.1 General construction rules

1. Use only data available before the forecast target.
2. Never use raw `secid` as a numeric feature.
3. Avoid ratios whose denominator can be near zero.
4. Express positive ratios as log differences, for example
   \[
   \log(a+\delta)-\log(b+\delta),
   \]
   with a documented positive offset `delta`.
5. Use differences rather than ratios for centered residual quantities.
6. Store every feature's formula, input window, and missing-value rule in a machine-readable feature manifest.

### 8.2 Block A: surface-coordinate features

These features distinguish nodes on the same underlying surface.

| Feature | Definition |
|---|---|
| `cp_call` | 1 for a call node, 0 otherwise. |
| `cp_put` | 1 for a put node, 0 otherwise. |
| `delta_abs` | Absolute delta in `[0,1]`. |
| `delta_signed` | Positive for calls and negative for puts. |
| `delta_distance_atm` | `abs(delta_abs - 0.50)`. |
| `delta_abs_sq` | `delta_abs^2`. |
| `log_days` | `log(days)`. |
| `days_scaled` | `(days-mean_grid_days)/sd_grid_days`. |
| `delta_x_log_days` | `delta_abs * log_days`. |
| `maturity_30`, ..., `maturity_365` | Six maturity one-hot indicators. |
| `delta_20`, ..., `delta_80` | Five absolute-delta one-hot indicators. |
| `short_maturity` | 1 if `days <= 60`. |
| `medium_maturity` | 1 if `91 <= days <= 182`. |
| `long_maturity` | 1 if `days >= 365`. |

### 8.3 Block B: underlying classification features

| Feature | Definition |
|---|---|
| `is_common_stock` | One if the issue type is ordinary/common equity. |
| `is_adr` | One if the issue is an ADR/ADS. |
| `is_etf_or_fund` | One for ETF, fund, or investment-trust issue types. |
| `is_index` | Vendor index indicator. |
| `sector_*` | One-hot indicators from the first digit or other coarse level of `industry_group`. |
| `sic_division_*` | Optional coarse SIC division indicators. Do not use hundreds of sparse four-digit SIC dummies. |
| `exchange_*` | Optional one-hot indicators for the active exchange at the end of inner training. |
| `listing_age_years` | Years from the first known listing date to the training cutoff. |

For the homogeneous common-equity primary panel, the asset-type indicators will be nearly constant and should be dropped automatically if their variance is zero.

### 8.4 Block C: underlying size, return, and realized-volatility features

All summaries use inner-training data only unless a recent window is explicitly stated.

| Feature | Definition |
|---|---|
| `log_price_median` | Median of `log(close_price)`. |
| `log_market_cap_median` | Median of `log(market_cap)`. |
| `log_share_volume_median` | Median of `log1p(underlying_volume)`. |
| `return_mean` | Mean vendor total return. |
| `return_sd` | Standard deviation of total return. |
| `return_downside_sd` | Root mean square of `min(return,0)`. |
| `return_skew` | Sample skewness of total return. |
| `return_excess_kurtosis` | Sample excess kurtosis, robustly capped. |
| `return_abs_q95` | 95th percentile of absolute return. |
| `return_acf_1` | Lag-1 autocorrelation of total return. |
| `return_acf_5` | Lag-5 autocorrelation. |
| `return_acf_22` | Lag-22 autocorrelation. |
| `rv20_mean` | Mean 20-trading-day realized volatility computed from returns. |
| `rv60_mean` | Mean 60-trading-day realized volatility. |
| `rv252_mean` | Mean 252-trading-day realized volatility. |
| `rv20_recent` | Realized volatility over the final 20 inner-training dates. |
| `rv60_recent` | Realized volatility over the final 60 inner-training dates. |
| `rv20_recent_logdiff` | `log(rv20_recent+eps)-log(rv20_mean+eps)`. |
| `market_return_beta` | OLS beta of the underlying return on the equal-weighted universe return, using training dates only. |

Use the vendor historical-volatility file for robustness, but do not mix vendor and self-computed realized-volatility definitions without clear labels.

### 8.5 Block D: option-liquidity and surface-quality features

| Feature | Definition |
|---|---|
| `log_option_volume_mean` | Mean of `log1p(option_volume_total)`. |
| `log_option_volume_median` | Median of `log1p(option_volume_total)`. |
| `log_option_volume_recent22` | Mean over the final 22 inner-training dates. |
| `option_volume_recent_logdiff` | Recent-22 mean minus long-run mean on the log1p scale. |
| `log_open_interest_mean` | Mean of `log1p(open_interest_total)`. |
| `log_open_interest_recent22` | Mean over the final 22 dates. |
| `open_interest_recent_logdiff` | Recent minus long-run mean on the log1p scale. |
| `put_call_volume_log_ratio` | `log(1+put volume)-log(1+call volume)`, averaged over training. |
| `call_volume_share` | Mean `call volume/(call volume+put volume+1)`. |
| `surface_coverage` | Fraction of selected node-date cells observed before imputation for the underlying. |
| `node_coverage` | Fraction observed before imputation for the exact component. |
| `interpolation_fraction` | Fraction of component dates filled by same-day surface interpolation. |
| `forward_fill_fraction` | Fraction filled by the maximum two-day carry-forward rule. |
| `dispersion_median` | Median valid surface dispersion for the component. |
| `dispersion_q90` | 90th percentile of valid dispersion. |
| `surface_age_days` | Trading days from first valid standardized-surface observation to training cutoff. |

### 8.6 Block E: node-level implied-volatility dynamics

Let `x_t = log_iv[t,k]` and `dx_t = x_t-x_{t-1}`.

| Feature | Definition |
|---|---|
| `logiv_mean` | Mean of `x_t`. |
| `logiv_sd` | Standard deviation of `x_t`. |
| `logiv_mad` | `1.4826 * median(abs(x_t-median(x_t)))`. |
| `logiv_q10` | 10th percentile of `x_t`. |
| `logiv_q50` | Median of `x_t`. |
| `logiv_q90` | 90th percentile of `x_t`. |
| `logiv_iqr` | 75th minus 25th percentile. |
| `logiv_acf_1` | Lag-1 autocorrelation of `x_t`. |
| `logiv_acf_5` | Lag-5 autocorrelation of `x_t`. |
| `logiv_acf_22` | Lag-22 autocorrelation of `x_t`. |
| `dlogiv_mean` | Mean of `dx_t`. |
| `dlogiv_sd` | Standard deviation of `dx_t`. |
| `dlogiv_mad` | Robust scale of `dx_t`. |
| `dlogiv_acf_1` | Lag-1 autocorrelation of `dx_t`. |
| `dlogiv_acf_5` | Lag-5 autocorrelation of `dx_t`. |
| `dlogiv_acf_22` | Lag-22 autocorrelation of `dx_t`. |
| `recent5_minus_long` | Mean of the last 5 `x_t` values minus the full-training mean. |
| `recent22_minus_long` | Mean of the last 22 values minus the full-training mean. |
| `recent63_minus_long` | Mean of the last 63 values minus the full-training mean. |
| `recent22_sd_logdiff` | `log(sd_last22+eps)-log(sd_full+eps)`. |
| `trend_slope_252` | OLS slope of `x_t` on scaled time over the last 252 dates. |
| `jump_fraction` | Fraction of `|dx_t|` exceeding three training MADs of `dx_t`. |

Do not form coefficients of variation or relative means from centered residuals.

### 8.7 Block F: underlying volatility-surface shape features

Compute a daily surface summary for each underlying and then average or robustly summarize it over inner training. Let `L(cp,d,m,t)` denote log IV for option type `cp`, absolute delta `d`, maturity `m`, and date `t`.

| Feature | Daily formula before training aggregation |
|---|---|
| `atm_level_m` | `0.5 * {L(C,0.50,m,t) + L(P,0.50,m,t)}` for `m` in `{30,91,182,365}`. |
| `term_slope_365_30` | `atm_level_365 - atm_level_30`. |
| `term_slope_182_30` | `atm_level_182 - atm_level_30`. |
| `put_skew_m` | `L(P,0.20,m,t) - L(P,0.50,m,t)` for `m` in `{30,91}`. |
| `call_skew_m` | `L(C,0.20,m,t) - L(C,0.50,m,t)` for `m` in `{30,91}`. |
| `smile_curvature_m` | `0.5*{L(P,0.20,m,t)+L(C,0.20,m,t)} - atm_level_m` for `m` in `{30,91}`. |
| `call_put_gap_m` | `L(C,0.50,m,t)-L(P,0.50,m,t)` for `m` in `{30,91}`. |
| `iv_minus_rv_m` | `atm_level_m - log(realized_volatility+eps)` for a matched horizon. |

For each daily quantity, create the training median, training standard deviation, and final-22-day mean minus the training median when sample size permits. These features are repeated for all 60 components belonging to the same underlying.

### 8.8 Block G: baseline-residual and predictive-dependence features

Construct these features after generating leakage-safe baseline errors. Let `r[t,k]` be the log-IV baseline error.

| Feature | Definition |
|---|---|
| `resid_sd` | Standard deviation of `r[t,k]`. |
| `resid_mad` | Robust residual scale. |
| `resid_acf_1` | Lag-1 residual autocorrelation. |
| `resid_acf_5` | Lag-5 residual autocorrelation. |
| `resid_acf_22` | Lag-22 residual autocorrelation. |
| `resid_recent_mean22` | Mean residual over the final 22 training dates. |
| `resid_recent_sd22_logdiff` | Log difference between recent and full residual scales. |
| `baseline_rmse` | Root mean squared historical baseline error. |
| `baseline_mae` | Mean absolute historical baseline error. |
| `market_resid_beta` | OLS beta of `r[t,k]` on the cross-sectional median residual at date `t`. |
| `sector_resid_beta` | OLS beta on the corresponding sector median residual. |
| `same_underlying_lag1_corr_mean` | Mean over the other 59 nodes of `corr(r[t,k], r[t-1,j])` within the same underlying. |
| `same_underlying_lag1_corr_max` | Maximum signed lag-one correlation over other nodes of the same underlying. |
| `same_underlying_lag1_corr_absmax` | Maximum absolute lag-one correlation over other nodes of the same underlying. |

The lag-one residual-correlation summaries are particularly important because they align the attention features with the cross-component quantity VARA is designed to exploit.

### 8.9 Feature preprocessing

Apply the following operations using inner-training information only:

1. Remove columns with zero or near-zero variance.
2. Replace nonfinite values with missing.
3. Winsorize continuous feature columns at the 0.5th and 99.5th cross-sectional percentiles.
4. Impute remaining feature missingness with the cross-sectional median. Add a missingness indicator for a feature when more than 0.5% of components were imputed.
5. Robustly standardize each continuous column:
   \[
   z_{kj}=\frac{x_{kj}-\operatorname{median}_k(x_{kj})}
   {1.4826\operatorname{MAD}_k(x_{kj})+10^{-8}}.
   \]
6. Keep binary one-hot variables in `{0,1}`.
7. Partition columns into feature blocks. Divide every block by the square root of its number of columns so that a large one-hot block does not dominate simply because it contains more columns:
   \[
   Z^{(g)}\leftarrow Z^{(g)}/\sqrt{q_g}.
   \]
8. Concatenate the blocks.
9. Normalize every component row to unit Euclidean norm:
   \[
   Z_k\leftarrow Z_k/\|Z_k\|_2.
   \]

Row normalization makes `Z[k]'Z[j]` a bounded cosine-type similarity and prevents high-norm components from becoming attention hubs.

Save:

```text
features_raw.parquet
features_processed.parquet
feature_manifest.csv
feature_preprocessor.json
```

---

## 9. VARA specification and estimation

Fit the existing VARA model to the standardized residual panel `z_t`:

\[
z_t=\operatorname{diag}(\alpha)z_{t-1}
+\operatorname{diag}(\beta)W(\gamma)z_{t-1}+\varepsilon_t.
\]

For `k != j`, use the paper's attention rule

\[
w_{kj}(\gamma)=
\frac{\exp\{\gamma\log(p) Z_k^\top Z_j\}}
{\sum_{\ell\ne k}\exp\{\gamma\log(p)Z_k^\top Z_\ell\}},
\qquad w_{kk}=0.
\]

Use the **two-stage estimator** because `p = 30,000` is much larger than `T = 1,941`:

1. Estimate the common attention parameter `gamma` by maximizing the concentrated quasi-likelihood.
2. Holding the estimated attention matrix fixed, estimate `(alpha_k, beta_k, sigma_k^2)` separately for every component.

Use the stationarity control already available in the VARA workflow. A defensible default is to constrain or project estimates so that

\[
|\alpha_k|+|\beta_k|\le 0.95,
\]

unless the existing implementation uses a more conservative preregistered bound.

### 9.1 Default tuning grid

Use rolling validation to select among:

```text
gamma initial grid: [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
ridge penalty:      [0, 1e-4, 1e-3, 1e-2, 1e-1]
response window:    [504, 756, 1260, all available dates]
feature sets:
  A = coordinate + metadata + underlying dynamics
  B = A + option liquidity + surface quality
  C = B + IV dynamics + surface shape
  D = C + baseline-residual predictive features
```

After the best grid point is found, use the existing local refinement procedure for `gamma` if available.

### 9.2 Numerical requirements

- Use a deterministic component order.
- Store the response panel in row-major or device-efficient format required by the implementation.
- Use `float32` for the large panel and attention matrix if supported; use higher precision for scalar likelihood comparisons when needed.
- Record CPU/GPU type, memory, worker count, wall time, and peak memory.
- Save random seeds even though the two-stage estimator is largely deterministic.
- Verify that every attention row sums to one within numerical tolerance and has a zero diagonal.

---

## 10. Forecast construction

### 10.1 Return the residual forecast to log-IV scale

If VARA forecasts the standardized residual as `z_hat[h,k]`, form

\[
\widehat r^{\text{VARA}}_{T+h,k}
=m_k+s_k\widehat z_{T+h,k}.
\]

### 10.2 Combine baseline and VARA

For horizon `h`, define

\[
\widehat y_{T+h,k}
=\widehat y^{\text{base}}_{T+h,k}
+\lambda_h\widehat r^{\text{VARA}}_{T+h,k}.
\]

Then return to implied-volatility units:

\[
\widehat{\text{IV}}_{T+h,k}=\exp(\widehat y_{T+h,k}).
\]

No zero truncation is needed because the exponential transformation guarantees positivity.

### 10.3 Shrinkage selection

Begin with one global `lambda` in `[0,1]`. For validation errors `e_i` from the baseline and candidate VARA corrections `c_i`, use the ridge-stabilized closed-form estimate

\[
\widehat\lambda=
\left[
\frac{\sum_i e_i c_i}
{\sum_i c_i^2+\tau}
\right]_0^1,
\]

where `tau` is chosen across rolling validation origins and `[.]_0^1` truncates to `[0,1]`.

Secondary specifications may use four horizon blocks:

```text
1-5, 6-10, 11-15, 16-22 trading days
```

or maturity groups:

```text
short: 30-60 days
medium: 91-182 days
long: 365 days
```

Do not estimate one independent `lambda_k` for every series from only 28 validation observations.

---

## 11. Forecasting protocols

### 11.1 Preferred protocol: rolling one-step-ahead

At each forecast date `t`:

1. Use observations through `t-1` only.
2. Recompute or update the baseline.
3. Append the newly observed historical baseline error when it becomes available.
4. Recompute all training-window transformations and features required by the selected specification.
5. Estimate VARA or update it according to a preregistered schedule.
6. Forecast `t`.
7. Store the forecast before reading the realized target.

For computational feasibility, VARA can be refitted every 5 trading days while the current fitted parameters are used on intervening dates. This changes the refit schedule, not the VARA model.

Use at least 126, preferably 252, independent daily forecast origins for formal predictive-accuracy tests.

### 11.2 M5-comparable protocol: fixed 28-day horizon

1. Select the specification using the 28-day validation block.
2. Refit the baseline, residual scaling, features, and VARA on all 1,913 pre-test dates.
3. Produce a recursive 28-trading-day forecast.
4. Keep all choices fixed throughout the test.
5. Report horizon-wise results for days 1 through 28 and aggregate results.

The one-step rolling experiment should be the main accuracy result; the fixed 28-day experiment should emphasize scalability and long-horizon behavior.

---

## 12. Evaluation metrics

Report metrics on both the log-IV and original-IV scales.

### 12.1 Point-forecast metrics

For matched forecast cells:

\[
\operatorname{RMSE}_{\log IV}
=\left\{N^{-1}\sum_i(\widehat y_i-y_i)^2\right\}^{1/2},
\]

\[
\operatorname{MAE}_{\log IV}=N^{-1}\sum_i|\widehat y_i-y_i|,
\]

\[
\operatorname{RMSE}_{IV}
=\left\{N^{-1}\sum_i(\widehat{IV}_i-IV_i)^2\right\}^{1/2}.
\]

Also report:

- Bias on the log-IV and IV scales.
- OOS `R^2` relative to the chosen univariate baseline.
- Percentage RMSE improvement over the baseline.
- Series-normalized RMSSE using a lag-one random-walk scale.

### 12.2 Segment results

Report accuracy by:

- Forecast horizon.
- Maturity.
- Absolute delta.
- Call versus put.
- Sector.
- Underlying option-liquidity quintile.
- Baseline-error volatility quintile.
- Surface-quality or dispersion quintile.

### 12.3 Predictive-accuracy inference

For rolling one-step forecasts, first average the loss over components within each forecast date. Apply the Diebold-Mariano test to the resulting time series of date-level loss differences. Do not treat millions of component-date cells as independent observations.

For only 28 forecast dates, label asymptotic p-values as exploratory and supplement them with block bootstrap confidence intervals over dates.

### 12.4 Probabilistic forecasts

If CRPS is reported, calibrate the variance of the **combined** forecast error using rolling validation. Do not automatically assume zero covariance between the baseline forecast error and the VARA residual-forecast error. A simple approach is to estimate a horizon-specific multiplicative variance calibration factor from validation residuals.

---

## 13. Required diagnostics

Before interpreting forecast gains, report the following.

| Diagnostic | Definition or purpose |
|---|---|
| Correction scale | `sd(vara_correction) / sd(baseline_error)` by horizon. |
| Correction alignment | Correlation between validation baseline error and VARA correction. |
| Incremental MSE identity | Compare `2*lambda*E[e*c]` with `lambda^2*E[c^2]`. |
| Attention entropy | `exp(-sum_j w[k,j] log w[k,j])`, the effective number of neighbors. |
| Top-neighbor composition | Fractions sharing the same underlying, sector, maturity, delta, and option type. |
| Attention hubs | Column sums of `W` and components receiving unusually large total attention. |
| Cross-term usefulness | Incremental `R^2` of `W z[t-1]` after controlling for `z[t-1,k]`. |
| Beta distribution | Distribution of estimated `beta_k`; many values near zero indicate weak cross-series signal. |
| Stability | Distribution of `abs(alpha_k)+abs(beta_k)`. |
| Missingness sensitivity | Results excluding imputed cells and high-dispersion nodes. |
| Oracle diagnostic | Training-only lag-one residual-correlation neighbors, used for diagnosis rather than final testing. |

The oracle diagnostic answers an important question: if a leakage-safe lagged-correlation attention matrix does not improve validation accuracy, then the financial panel contains little exploitable lag-one cross-series signal. If the oracle improves but the feature-based attention does not, the problem is feature alignment rather than the VARA dynamics.

---

## 14. Recommended ablation sequence

Run the following experiments in order so that each change has an interpretable effect.

| Order | Experiment |
|---:|---|
| 1 | HAR baseline only. |
| 2 | VARA on centered but unscaled HAR residuals. |
| 3 | VARA on robustly standardized HAR residuals. |
| 4 | Add coordinate and underlying metadata features. |
| 5 | Add option liquidity and surface-quality features. |
| 6 | Add node IV-dynamics and surface-shape features. |
| 7 | Add baseline-residual predictive features. |
| 8 | Tune recent response-window length. |
| 9 | Tune global or horizon-block shrinkage. |
| 10 | Repeat using AutoETS residuals for direct comparison with M5. |
| 11 | Robustness using quote-derived IV from actual liquid options. |

Lock the final strategy before evaluating the untouched final test.

---

## 15. Output files and directory structure

Use a structure such as:

```text
project/
  config/
    data_snapshot.yaml
    schema_mapping.yaml
    experiment.yaml
  data/
    raw/                    # licensed; do not redistribute
    interim/
      candidate_universe.parquet
      selected_universe.parquet
      surface_clean_long.parquet
    processed/
      component_map.parquet
      log_iv_panel.zarr
      baseline_errors.zarr
      residual_scalers.parquet
      features_raw.parquet
      features_processed.parquet
      feature_manifest.csv
      split_dates.json
  models/
    baseline/
    vara/
  forecasts/
    validation_forecasts.parquet
    test_forecasts.parquet
  results/
    aggregate_metrics.csv
    segment_metrics.csv
    attention_diagnostics.csv
    runtime.json
    figures/
  logs/
```

Minimum columns in the final forecast file:

```text
origin_date
target_date
horizon
component_index
component_id
observed_log_iv
observed_iv
baseline_log_iv_hat
vara_resid_hat_std
vara_correction
lambda
final_log_iv_hat
final_iv_hat
is_target_observed
was_input_imputed
```

---

## 16. Extraction template

The exact table names must be obtained from the current WRDS or OptionMetrics dictionary. The logical query is:

```sql
SELECT
    secid,
    date,
    days,
    delta,
    cp_flag,
    impl_volatility,
    impl_strike,
    impl_premium,
    dispersion
FROM <optionmetrics_schema>.<volatility_surface_table>
WHERE date BETWEEN :start_date AND :end_date
  AND days IN (30, 60, 91, 122, 182, 365)
  AND ABS(delta_normalized) IN (0.20, 0.35, 0.50, 0.65, 0.80)
  AND cp_flag IN ('C', 'P');
```

Extract the security master, name history, security price, option volume, and historical volatility for the same candidate `secid` values and date range. Use server-side filters and partition downloads by year when necessary.

---

## 17. End-to-end pseudocode

```python
# 1. Freeze data snapshot and map vendor columns to canonical names.
raw = extract_optionmetrics(snapshot_config, schema_mapping)

# 2. Normalize identifiers, dates, delta units, IV units, and missing sentinels.
surface = clean_volatility_surface(raw.surface)
metadata = build_asof_security_metadata(raw.security, raw.security_names)
prices = clean_security_prices(raw.security_prices)
liquidity = clean_option_volume(raw.option_volume)

# 3. Define 1,941 trading dates and inner-train/validation/test splits.
splits = make_fixed_date_splits(surface.dates, total=1941, validation=28, test=28)

# 4. Rank candidates using inner-training coverage and liquidity only.
universe_rank = rank_underlyings(surface, liquidity, splits.inner_train)
selected_secids = select_top_eligible(universe_rank, n=500)

# 5. Build the fixed 60-node panel and leakage-safe missing-data flags.
iv_panel, quality_flags = build_surface_panel(
    surface,
    selected_secids,
    maturities=[30, 60, 91, 122, 182, 365],
    deltas=[0.20, 0.35, 0.50, 0.65, 0.80],
    cp_flags=['C', 'P'],
    same_day_interpolation=True,
    max_forward_fill=2,
)

# 6. Transform to log IV and generate leakage-safe baseline forecasts/errors.
log_iv = np.log(iv_panel)
baseline_forecasts, baseline_errors = rolling_har_errors(log_iv, splits)

# 7. Robustly standardize baseline errors by component.
z_panel, residual_scalers = robust_standardize_residuals(
    baseline_errors,
    training_dates=splits.inner_train,
)

# 8. Construct component features from inner-training information only.
Z_raw = build_component_features(
    component_map,
    log_iv,
    baseline_errors,
    metadata,
    prices,
    liquidity,
    quality_flags,
    training_dates=splits.inner_train,
)
Z = preprocess_features(Z_raw, fit_dates=splits.inner_train)

# 9. Tune feature set, response window, ridge, gamma, and lambda on validation.
best_config = rolling_validation_select(
    z_panel=z_panel,
    Z=Z,
    baseline_forecasts=baseline_forecasts,
    validation_dates=splits.validation,
)

# 10. Recompute all training-only objects through the final training date.
final_inputs = rebuild_through_training_end(best_config, splits.final_training)

# 11. Fit the unchanged two-stage VARA estimator.
vara_fit = fit_vara_two_stage(final_inputs.z_panel, final_inputs.Z, best_config)

# 12. Forecast standardized residuals, invert scaling, and shrink the correction.
vara_z_hat = forecast_vara(vara_fit, horizon=len(splits.test))
correction = invert_residual_scaling(vara_z_hat, final_inputs.residual_scalers)
final_log_iv_hat = final_inputs.baseline_test_hat + best_config.lambda_ * correction
final_iv_hat = np.exp(final_log_iv_hat)

# 13. Evaluate only the untouched test and save diagnostics.
results = evaluate_and_diagnose(final_iv_hat, final_log_iv_hat, splits.test)
save_reproducible_outputs(results, best_config, snapshot_config)
```

---

## 18. Stop conditions and validation checks

Do not run the expensive full-panel VARA fit until all of the following pass:

```text
[ ] Exactly 500 selected underlyings.
[ ] Exactly 60 ordered nodes per underlying.
[ ] Exactly 30,000 component IDs.
[ ] Exactly 1,941 ordered trading dates for the M5-comparable experiment.
[ ] No duplicate component-date keys.
[ ] No future-based backfilling or feature calculation.
[ ] No missing or infinite model-input values.
[ ] Every continuous feature is finite after preprocessing.
[ ] Every feature row has unit norm.
[ ] Similarities lie within numerical tolerance of [-1, 1].
[ ] Every attention row sums to one and has zero self-weight.
[ ] Universe selection uses inner training only.
[ ] Validation choices are locked before test evaluation.
[ ] Forecast file exists before test outcomes are joined.
[ ] Matched-cell count and missing-target count are reported.
```

---

## 19. Suggested application write-up

A concise methods description can follow this template:

> We construct a daily panel from the OptionMetrics IvyDB US standardized implied-volatility surface. Each component corresponds to one underlying-security, option-type, delta, and constant-maturity node. We select 500 liquid underlyings using only the inner-training period and retain 60 surface nodes per underlying, yielding 30,000 series observed over 1,941 trading days. The forecasting response is the robustly standardized historical forecast error from a componentwise HAR model for log implied volatility. Component features describe surface location, security classification, underlying return and volatility dynamics, option-market liquidity, surface quality, historical surface shape, and baseline-residual predictability. Continuous features are robustly standardized, feature blocks are balanced, and each component vector is normalized to unit length before constructing the VARA attention matrix. The common attention parameter is estimated by the concentrated quasi-likelihood and the component-specific coefficients are estimated in the second stage. All universe choices, transformations, features, and tuning parameters are constructed without using final-test observations.

---

## 20. References

1. OptionMetrics. **IvyDB US: Historical Options and Implied Volatility Data**. <https://optionmetrics.com/united-states/>
2. Wharton Research Data Services. **OptionMetrics data vendor page**. <https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/optionmetrics/>
3. OptionMetrics. **IvyDB US Reference Manual, archived version 5.0**. <https://lib.tsinghua.edu.cn/PDF/OptionMetricsshiyongzhinan.pdf>
4. Gao, Y., Li, F., Zhu, X., and Wang, H. **Vector Autoregression with Attention for High-Dimensional Time-Series**, manuscript dated July 16, 2026.

