# Suggestions to Improve the M5 Data Application of VARA

## Scope

These recommendations are intended to improve the Walmart M5 forecasting application **without changing the implementation of VARA itself**. The VARA model equation, attention parameterization, two-stage estimator, likelihood, and optimization code can remain unchanged. The proposed changes concern only:

- construction and scaling of the response panel;
- construction of component-level features;
- organization of the M5 series into panels;
- choice of estimation windows;
- validation and forecast-combination procedures;
- baseline forecasting models; and
- probabilistic forecast calibration.

The current M5 application is valuable as a scalability demonstration: it fits VARA to 30,490 item-store series and produces a small RMSE improvement over AutoETS. However, the gain is very small. The fixed-shrinkage specification reduces RMSE from 1.980 to 1.978, and the validation-selected specification reduces it to 1.979. AutoETS also remains slightly better in CRPS. These results indicate that the VARA correction is either too small or insufficiently aligned with future AutoETS forecast errors.

Ignoring truncation at zero, the final forecast is

\[
\widehat Y_{t+h,k}
=
\widehat Y^{\mathrm{AutoETS}}_{t+h,k}
+
\lambda_h\widehat r^{\mathrm{VARA}}_{t+h,k}.
\]

Let

\[
e_{t+h,k}=Y_{t+h,k}-\widehat Y^{\mathrm{AutoETS}}_{t+h,k}
\]

be the AutoETS forecast error and let

\[
c_{t+h,k}=\widehat r^{\mathrm{VARA}}_{t+h,k}
\]

be the VARA correction. The reduction in mean squared error is approximately

\[
2\lambda E(ec)-\lambda^2E(c^2).
\]

Therefore, almost no improvement means that the correction is small, weakly correlated with the future baseline error, excessively noisy, or some combination of the three.

---

## 1. Standardize the AutoETS residual panel before fitting VARA

### Current concern

The M5 application fits VARA to a centered AutoETS residual panel, but it does not appear to scale each residual series to a comparable variance. Centering alone is insufficient because item-store series differ substantially in sales volume, intermittency, and residual variance.

The cross-series predictor in VARA is

\[
W r_{t-1}.
\]

If residuals remain on their original sales scales, a high-volume series can dominate the attention-weighted average for a low-volume target. The component-specific coefficient \(\beta_k\) can rescale the final regressor, but it cannot undo the fact that a few high-variance sources may dominate a mixture of otherwise useful neighbors.

### Recommendation

For each item-store series, construct a standardized residual panel

\[
z_{t,k}=\frac{r_{t,k}-\mu_k}{s_k},
\]

where \(\mu_k\) and \(s_k\) are calculated from the training data only. A robust scale is preferable because M5 residuals can contain large promotion, stockout, and event-related errors. Suitable choices include:

- median absolute deviation multiplied by 1.4826;
- residual root mean square;
- RMSSE seasonal scale; or
- a hybrid scale with a small positive floor.

The standardized residuals can optionally be winsorized at a high training-only quantile. Fit the existing VARA implementation to \(z_t\), forecast \(\widehat z_{t+h,k}\), and transform the correction back to the original scale:

\[
\widehat r_{t+h,k}=s_k\widehat z_{t+h,k}.
\]

### Expected benefit

This change makes cross-series residual information comparable and should improve the stability and interpretability of the attention-weighted lag. It is the highest-priority preprocessing experiment because it requires no change to VARA and directly addresses a scale mismatch that is especially severe in retail panels.

---

## 2. Construct attention features from raw demand rather than centered residual ratios

### Current concern

The main M5 attention features are constructed from AutoETS residuals. Several of the features divide by the residual mean, including:

- coefficient of variation;
- recent relative means;
- half-sample trend ratios;
- day-of-week relative means;
- weekend ratios;
- event ratios; and
- SNAP ratios.

AutoETS residuals are designed to have a mean close to zero. Dividing by a mean that is near zero can create very large and unstable values. Setting the ratio to zero only when the denominator is exactly zero does not solve the problem when the denominator is merely small.

The residual-based intermittency features are also unlikely to be informative. A residual is classified as nonzero whenever its absolute value exceeds \(10^{-12}\). Since fitted residuals are usually continuous and almost never exactly zero, the zero and nonzero fractions will often be nearly degenerate.

### Recommendation

Keep AutoETS residuals as the response modeled by VARA, but construct the component features \(Z_k\) primarily from **raw demand and metadata**. The response panel and the attention-feature panel do not need to use the same transformation.

Use raw-demand features for:

- demand level and variation;
- zero-demand frequency and inter-demand intervals;
- recent demand state;
- day-of-week demand profile;
- event response;
- SNAP response;
- price sensitivity; and
- release and availability history.

Replace unstable ratios such as \(a/b-1\) with regularized transformations. For positive raw-demand quantities, a stable alternative is

\[
\log\left(\frac{a+\delta}{b+\delta}\right),
\]

where \(\delta>0\) is selected from the training demand scale. Robust differences, standardized contrasts, or rank-based summaries can also replace ratios.

### Expected benefit

Raw-demand features more reliably characterize item-store similarity, while the residual response remains focused on correcting the baseline forecast. This separation should reduce feature noise and improve the alignment between the attention matrix and genuine retail similarity.

---

## 3. Robustly standardize, balance, and row-normalize the component features

### Current concern

The M5 feature matrix contains 66 columns from demand, calendar, price, release, and hierarchy groups. The columns are standardized across series, but the reported construction does not indicate that each row is normalized to unit length.

Since attention similarity is based on the inner product

\[
s_{kj}=Z_k^\top Z_j,
\]

variation in the row norm \(\|Z_k\|_2\) can create artificial attention hubs. A component with an unusually large feature norm may appear similar to many other components even when the feature directions are not closely aligned.

Feature groups also have unequal dimensions. The hierarchy group contains many indicator columns and can dominate smaller groups merely because it has more coordinates.

### Recommendation

Apply the following feature-processing sequence using training data only:

1. Clip or winsorize heavy-tailed continuous features.
2. Robustly center and scale each continuous feature.
3. Scale each feature group by \(1/\sqrt{q_g}\), where \(q_g\) is the number of variables in group \(g\).
4. Normalize each item-store feature vector to unit Euclidean norm:

\[
Z_k\leftarrow\frac{Z_k}{\|Z_k\|_2}.
\]

After row normalization, the inner product becomes cosine similarity. This makes the empirical construction more comparable to the simulation design, in which feature vectors are normalized.

### Expected benefit

This should reduce the influence of outlying feature magnitudes, prevent hierarchy indicators from overwhelming smaller but predictive feature groups, and make the global attention-strength parameter easier to interpret.

---

## 4. Add item identity and residual-dynamics features

### Current concern

The hierarchy indicators include state, store, category, and department, but not item identity. Yet the same product sold in different stores is one of the most natural sources of cross-series information in M5.

The current features mainly describe product level and historical calendar response. They do not directly describe whether one series has residual dynamics similar to another series, even though VARA is fitted to residuals.

### Recommendation

Add features that are likely to be associated with predictable cross-series residual dependence:

- item identity, a compact item embedding, or a hashed item indicator;
- same-item-across-store demand summaries;
- residual autocorrelations at lags 1, 7, 14, and 28;
- raw-demand autocorrelations and seasonal strength;
- a seven-dimensional day-of-week demand profile;
- event- and SNAP-response profiles;
- correlation with store, department, category, and state aggregates;
- recent versus long-run residual variance;
- release age and active-history fraction;
- recent price change and promotion frequency; and
- stockout-like or prolonged-zero indicators.

All dynamic features must be computed using only observations available before the forecasting origin.

### Expected benefit

These additions would make the attention matrix more relevant to the specific residual dependence that VARA is intended to exploit, rather than only to broad descriptive similarity.

---

## 5. Fit the unchanged VARA model to economically homogeneous subpanels

### Current concern

The current application constructs attention over all 30,490 item-store series. In such a large and heterogeneous panel, a single global attention-strength parameter can lead to either diffuse or unstable attention.

If many products have similar coarse features, each row may distribute weight across hundreds or thousands of weakly related series. Because the residuals are centered, the resulting weighted average can approach zero. Conversely, a concentrated softmax may select accidental neighbors based on small, noisy feature differences.

### Recommendation

Run the same VARA implementation separately on smaller, economically coherent panels. Candidate partitions include:

- store by department;
- state by department;
- store by category;
- same item across stores;
- department by demand-intermittency class; and
- department by release-age or sales-volume group.

A particularly useful two-view design is:

1. a **same-item-across-stores** VARA model; and
2. a **same-store-within-department** VARA model.

Their residual forecasts can be combined outside VARA using validation weights.

### Expected benefit

Local panels reduce attention dilution, give the global \(\gamma\) parameter a more homogeneous population, and improve economic interpretability. They also reduce computation and permit more extensive rolling validation.

---

## 6. Train VARA on genuine historical forecast errors

### Current concern

The M5 specification uses residuals from an AutoETS fit, while the Tourism application uses genuine historical one-step-ahead forecast errors. In-sample residuals and future forecasting errors need not have the same distribution.

A model fitted to in-sample residuals may learn patterns that do not persist when AutoETS is used recursively out of sample. The stronger Tourism result suggests that aligning the VARA target with the error it will actually predict may be important.

### Recommendation

Generate a leakage-safe panel of historical AutoETS forecast errors using rolling or expanding origins. Practical computational compromises include:

- refitting AutoETS every 7 or 28 days rather than every day;
- using state-space one-step-ahead innovations with parameters estimated only from past data;
- producing one-step-ahead errors from periodic expanding-window refits; or
- limiting the VARA training sample to the most recent 364, 728, or 1,092 days.

For every forecasting origin, all residual scaling, feature construction, and model fitting must use only previously observed information.

### Expected benefit

This should improve the match between the VARA training target and the final correction target. A recent estimation window may also adapt better to changes in item availability, pricing, promotions, and demand regimes.

---

## 7. Use weekly structure without changing the lag-one VARA implementation

### Current concern

Daily M5 demand has strong weekly seasonality. The current VARA model uses lag one, while useful cross-series dependence may occur at lag seven rather than lag one.

### Recommendation

Create seven day-of-week subsequences. For each weekday, retain only observations from that weekday and fit the unchanged lag-one VARA model to the resulting weekly sequence. In this representation, VARA lag one corresponds to a seven-day calendar lag.

For a 28-day forecast, use the model associated with each forecast date's weekday. Related external partitions can be created for:

- SNAP versus non-SNAP dates;
- event versus non-event dates; and
- regular versus promotion periods.

These are separate applications of the same VARA code, not changes to its implementation.

### Expected benefit

This provides a direct way to exploit weekly residual dependence while retaining the existing first-order VARA model.

---

## 8. Replace the single validation block with repeated rolling origins

### Current concern

The scalar shrinkage parameter and other choices are selected using one 28-day validation period. A single month can be highly dependent on the particular events, prices, SNAP schedule, and product availability during that period.

The fixed value \(\lambda=0.4\) performs slightly better on the final test period than the validation-selected value \(\lambda=0.632\). This suggests that selection from a single validation block may be unstable or that the validation objective is nearly flat.

### Recommendation

Use several historical 28-day rolling-origin blocks to select:

- residual scaling method;
- feature subset;
- row normalization and group weighting;
- panel partition;
- estimation-window length;
- ridge regularization;
- attention-strength search range; and
- residual-combination weights.

Four to eight rolling origins would provide a more reliable compromise between computational cost and selection stability.

### Expected benefit

Repeated validation should reduce sensitivity to a single month and produce tuning choices that generalize better to the final test period.

---

## 9. Use structured shrinkage rather than one global coefficient

### Current concern

The M5 application uses one scalar \(\lambda\) for all 30,490 series and all 28 horizons. This imposes strong pooling across products with very different demand properties and across horizons with different forecast uncertainty.

### Recommendation

Estimate a small number of pooled shrinkage coefficients outside VARA. Reasonable groupings include:

- horizons 1-7, 8-14, 15-21, and 22-28;
- department;
- dense, moderately intermittent, and highly intermittent demand;
- sales-volume bins; or
- department by horizon block.

A regularized estimate for group \(g\) is

\[
\widehat\lambda_g
=
\left[
\frac{\sum_{(h,k)\in g}e_{h,k}c_{h,k}}
{\sum_{(h,k)\in g}c_{h,k}^2+\tau}
\right]_0^1,
\]

where \(\tau\) is selected over rolling origins and \([\cdot]_0^1\) truncates the result to the interval \([0,1]\).

Avoid estimating a separate \(\lambda_k\) for every item-store series because 28 validation observations per series are insufficient.

### Expected benefit

Structured shrinkage allows VARA to contribute more where its correction is predictive and less where it is noisy, while retaining enough pooling for stable estimation.

---

## 10. Use a baseline that incorporates known future M5 covariates

### Current concern

The M5 calendar and price files contain future information such as day of week, events, SNAP status, and selling price. In the current application, these variables mainly affect static attention features. They do not directly tell the forecast model that a particular future date is an event date, SNAP date, or price-change period.

AutoETS does not directly use these future regressors. Consequently, VARA is asked to correct predictable calendar and price effects after the fact, even though a lag-one residual model may not be well suited to that task.

### Recommendation

Use a stronger univariate or global baseline that incorporates known future covariates, such as:

- dynamic regression with seasonal errors;
- regression plus ETS or ARIMA errors;
- a tree-based retail forecasting baseline;
- a generalized linear demand model; or
- a forecast combination of AutoETS and a covariate-aware model.

Then apply the same VARA implementation to the resulting residual or historical forecast-error panel.

### Expected benefit

This should improve the final forecasting system by removing known calendar and price effects before VARA models the remaining cross-series dependence. The incremental percentage attributed solely to VARA may become smaller because the residual target is harder, but the total forecast should improve.

---

## 11. Calibrate the combined predictive variance separately

### Current concern

The final predictive variance is approximated by adding the baseline variance and the scaled VARA residual variance while setting their covariance to zero:

\[
\widehat v^{\mathrm{final}}
\approx
\widehat v^{\mathrm{baseline}}
+
\lambda^2\widehat v^{\mathrm{VARA-resid}}.
\]

Because VARA is trained on baseline residuals, the baseline forecast error and the VARA correction error are not necessarily independent. The zero-covariance approximation may partly explain why AutoETS has a slightly better CRPS despite VARA's small RMSE advantage.

### Recommendation

Calibrate the variance of the combined forecast directly from rolling validation errors. Options include:

- estimating the omitted covariance term;
- applying a horizon-specific multiplicative calibration;
- estimating variance by horizon and department;
- estimating variance by demand-intermittency class; or
- minimizing validation CRPS over a small set of variance multipliers.

For example,

\[
\widehat v^{\mathrm{cal}}_{h,k}
=
 a_h\widehat v^{\mathrm{plug-in}}_{h,k},
\]

where \(a_h\) is selected on rolling validation origins.

### Expected benefit

This can improve probabilistic calibration without altering VARA point estimation.

---

## Diagnostics to run before another full-panel fit

The following diagnostics can determine whether the main limitation is weak cross-series signal, poor feature alignment, excessive attention diffusion, or inappropriate shrinkage.

| Diagnostic | Interpretation |
|---|---|
| Standard deviation of the VARA correction divided by the standard deviation of the AutoETS error | Determines whether the correction is too small to matter |
| Correlation between the validation AutoETS error and the VARA correction | Measures directional alignment of the correction |
| Validation MSE as a function of \(\lambda\) | Reveals whether the combination objective is flat or has a stable optimum |
| Incremental \(R^2\) of \(Wr_{t-1}\) after controlling for \(r_{t-1,k}\) | Measures usable cross-series residual signal |
| Attention entropy or effective neighbor count | Distinguishes diffuse attention from excessive concentration |
| Fraction of top neighbors sharing item, store, department, and category | Checks economic plausibility of the learned attention |
| Distribution of estimated \(\widehat\beta_k\) | Shows whether VARA effectively reverts to univariate dynamics for most series |
| Performance by department, store, volume, intermittency, and horizon | Identifies subgroups where VARA is useful |
| Oracle training-only lagged-residual attention | Distinguishes weak cross-series signal from poor feature construction |

A useful oracle diagnostic is to construct an attention matrix from training-only lagged residual correlations or predictive scores. This oracle should not be used for the final test evaluation unless it is constructed in a fully leakage-safe rolling procedure. Its purpose is diagnostic:

- If the oracle attention also gives no validation improvement, the M5 residual panel may contain little exploitable lag-one cross-series predictability.
- If the oracle improves performance but the domain-feature attention does not, the main problem is feature construction or panel organization.

---

## Recommended experiment sequence

A staged experiment plan can isolate the main sources of weak performance while controlling computational cost.

### Stage 1: Low-cost preprocessing ablations

1. Current M5 specification.
2. Robust residual standardization only.
3. Raw-demand attention features instead of residual-derived ratios.
4. Robust feature clipping, group balancing, and row normalization.
5. Add item identity and residual-dynamics features.

These experiments can first be run on selected stores or departments before scaling to the full panel.

### Stage 2: Panel and temporal design

6. Store-department local panels.
7. Same-item-across-stores panels.
8. Recent VARA estimation windows of 364, 728, and 1,092 days.
9. Genuine rolling AutoETS forecast errors.
10. Day-of-week subsequence VARA.

### Stage 3: Validation and combination

11. Repeated 28-day rolling-origin validation.
12. Horizon-block shrinkage.
13. Department- or intermittency-specific shrinkage.
14. External combination of multiple local-panel VARA corrections.

### Stage 4: Final system improvement

15. Covariate-aware baseline using event, SNAP, calendar, and price information.
16. Direct calibration of the combined predictive variance.
17. Final full-panel or multi-panel refit using the best validated configuration.

---

## Highest-priority recommendations

If only a small number of changes can be tested, prioritize the following:

1. **Robustly standardize the AutoETS residual panel.**
2. **Construct attention features from raw demand rather than centered residual ratios.**
3. **Clip, balance, and row-normalize the feature vectors.**
4. **Add item identity and residual-dynamics features.**
5. **Fit VARA within homogeneous local panels.**
6. **Use repeated rolling-origin validation and structured shrinkage.**

These changes address the most likely reasons that the current correction is small or weakly aligned with future baseline errors, while preserving the existing VARA implementation.

---

## Suggested interpretation for the paper

> The M5 application primarily demonstrates that VARA can be estimated on a panel of 30,490 item-store series. Its incremental improvement over AutoETS is modest, which is consistent with weak recoverable lag-one cross-series dependence after univariate filtering and with imperfect alignment between static domain-feature similarity and predictive residual dependence. Several aspects of the initial application may further attenuate the signal: the residual series are centered but not scale-standardized, several attention features use ratios whose denominators are near-zero residual means, and a full-panel softmax must identify useful neighbors among more than 30,000 heterogeneous series. A stronger application therefore standardizes the residual response, constructs stable attention features from raw demand and item-level information, estimates VARA within economically homogeneous subpanels, uses leakage-safe historical forecast errors, and selects forecast-combination weights over multiple rolling validation origins.

