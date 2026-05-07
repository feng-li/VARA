import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy
from tqdm import tqdm

try:
    from ._bootstrap import RESULTS_DIR
except ImportError:
    from _bootstrap import RESULTS_DIR

from vara.run_config import load_config_json_defaults, save_run_config, select_config_values


AR1_CONFIG_KEYS = (
    "seed",
    "R",
    "T",
    "phi_true",
    "sigma_true",
    "alpha",
    "results_dir",
)


def _load_config_json_defaults(path):
    return load_config_json_defaults(path, AR1_CONFIG_KEYS)


def run_ar1_coverage(
    seed=None,
    R=10_000,
    T=1000,
    phi_true=0.6,
    sigma_true=1.0,
    alpha=0.1,
    results_dir=RESULTS_DIR,
):
    if seed is not None:
        np.random.seed(seed)

    cover_count = 0
    interval_lengths = []

    for _ in tqdm(range(R), desc="Simulating AR(1) coverage"):
        y = np.zeros(T)
        eps = np.random.normal(0, sigma_true, size=T)
        for t in range(1, T):
            y[t] = phi_true * y[t - 1] + eps[t]

        y_lag = y[:-1]
        y_curr = y[1:]
        phi_hat = np.sum(y_lag * y_curr) / np.sum(y_lag ** 2)

        residuals = y_curr - phi_hat * y_lag
        sigma2_hat = np.sum(residuals ** 2) / (T - 2)
        se_phi = np.sqrt(sigma2_hat / np.sum(y_lag ** 2))

        z = scipy.stats.norm.ppf(1 - alpha / 2.0)
        lower = phi_hat - z * se_phi
        upper = phi_hat + z * se_phi
        interval_lengths.append(upper - lower)

        if lower <= phi_true <= upper:
            cover_count += 1

    ecp = cover_count / R
    avg_len = float(np.mean(interval_lengths))
    nominal = 1 - alpha

    print(f"Empirical coverage probability: {ecp:.3f}")
    print(f"Average CI length: {avg_len:.3f}")
    print(f"Nominal level: {nominal:.2f}")

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_tag = datetime.now().strftime("%Y%m%d_%H.%M.%S")
    config_tag = f"R={R}_T={T}_seed={seed}_{run_tag}"
    out_path = results_dir / f"results_ecp_ar1_{config_tag}.csv"
    config_path = results_dir / f"config_ecp_ar1_{config_tag}.json"

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "R",
                "T",
                "phi_true",
                "sigma_true",
                "alpha",
                "nominal",
                "coverage",
                "avg_len",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "seed": seed,
            "R": R,
            "T": T,
            "phi_true": phi_true,
            "sigma_true": sigma_true,
            "alpha": alpha,
            "nominal": nominal,
            "coverage": ecp,
            "avg_len": avg_len,
        })

    run_config = select_config_values(locals(), AR1_CONFIG_KEYS)
    run_config.update({
        "command": " ".join(sys.argv),
        "config_tag": config_tag,
        "run_tag": run_tag,
        "out_path": str(out_path),
    })
    save_run_config(config_path, run_config)

    print(f"Saved run config to: {config_path}")
    print(f"Saved results to: {out_path}")
    return ecp, avg_len


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AR(1) empirical coverage simulation.")
    parser.add_argument(
        "--config-json",
        type=str,
        default=None,
        help="Optional saved run-config JSON. Explicit CLI arguments override JSON defaults.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--R", type=int, default=10_000)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--phi-true", type=float, default=0.6)
    parser.add_argument("--sigma-true", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(RESULTS_DIR),
        help="Directory for CSV and config outputs.",
    )

    config_args, _ = parser.parse_known_args()
    if config_args.config_json:
        parser.set_defaults(**_load_config_json_defaults(config_args.config_json))
    args = parser.parse_args()

    run_ar1_coverage(
        seed=args.seed,
        R=args.R,
        T=args.T,
        phi_true=args.phi_true,
        sigma_true=args.sigma_true,
        alpha=args.alpha,
        results_dir=Path(args.results_dir),
    )
