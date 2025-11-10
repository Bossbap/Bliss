#!/usr/bin/env python3
"""
Grid-search hyper-parameters for Bliss regressors, now parallelised.

Usage example:
  # 8-core parallel run, 2-fold CV
  python hyper_param_search.py \
      --data_dir thirdparty/bliss/regressor_test/datasets/femnist \
      --model xgboost --regressor g \
      --n_jobs 8 --cv_splits 2
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, ParameterGrid

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None  # type: ignore


def resolve_xgb_device(preferred: str | None) -> tuple[str, str, str]:
    """
    Return a tuple (device, tree_method, predictor) compatible with XGBoost ≥3.1.

    • If `preferred` is provided, use it verbatim (with CPU/GPU-specific defaults).
    • Otherwise, probe for a CUDA GPU once; fall back to CPU on failure.
    """
    if preferred:
        device = preferred
        is_cpu = device.lower() == "cpu"
        return (
            device,
            "hist" if is_cpu else "gpu_hist",
            "auto" if is_cpu else "gpu_predictor",
        )

    try:
        import xgboost as xgb

        probe = xgb.XGBRegressor(
            n_estimators=1,
            max_depth=1,
            tree_method="gpu_hist",
            device="cuda:0",
            objective="reg:squarederror",
            eval_metric="rmse",
        )
        probe.fit(
            np.zeros((4, 1), dtype=np.float32),
            np.zeros(4, dtype=np.float32),
            verbose=False,
        )
        return "cuda:0", "gpu_hist", "gpu_predictor"
    except Exception:
        return "cpu", "hist", "auto"


def load_latest_csv(data_dir: Path, regressor: str) -> pd.DataFrame:
    csvs = sorted(data_dir.glob(f"{regressor}_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No {regressor}_*.csv in {data_dir}")
    logging.info("Dataset: %s", csvs[-1].name)
    return pd.read_csv(csvs[-1])


def eval_cfg(model_name: str,
             params: dict,
             X: np.ndarray,
             y: np.ndarray,
             rounds: np.ndarray,
             cv: int,
             rng_seed: int = 42) -> float:
    """Return unweighted mean RMSE across rounds, with early stopping."""
    is_xgb = (model_name == "xgboost")
    if is_xgb:
        from xgboost import XGBRegressor
    else:
        from sklearn.neural_network import MLPRegressor

    rmses: list[float] = []
    for r in np.unique(rounds):
        mask = (rounds == r)
        X_r, y_r = X[mask], y[mask]
        if len(X_r) < cv:
            continue

        kf = KFold(n_splits=cv, shuffle=True, random_state=rng_seed)
        se: list[float] = []

        for tr, va in kf.split(X_r):
            if is_xgb:
                model = XGBRegressor(
                    **params,
                    early_stopping_rounds=10
                )
                model.fit(
                    X_r[tr], y_r[tr],
                    eval_set=[(X_r[va], y_r[va])],
                    verbose=False
                )
            else:
                es_kwargs = dict(
                    early_stopping=True,
                    n_iter_no_change=10,
                    validation_fraction=0.1
                )
                model = MLPRegressor(
                    **params,
                    random_state=rng_seed,
                    **es_kwargs
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    model.fit(X_r[tr], y_r[tr])
            y_hat = model.predict(X_r[va])
            se.append(mean_squared_error(y_r[va], y_hat))

        rmses.append(np.sqrt(np.mean(se)))

    return float(np.mean(rmses))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,
                        help="…/regressor_test/datasets/<job_name>")
    parser.add_argument("--model", required=True,
                        choices=["xgboost", "mlp"])
    parser.add_argument("--regressor", required=True,
                        choices=["g", "h"])
    parser.add_argument("--cv_splits", type=int, default=2,
                        help="Number of CV folds (default=2 for speed)")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Number of parallel configs to evaluate")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="XGBoost device string (e.g. cuda:0 or cpu). Auto-detect when omitted.",
    )
    parser.add_argument(
        "--max_configs",
        type=int,
        default=None,
        help="Optional cap on number of configs to evaluate (sampled from the grid)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(Path("thirdparty/bliss/regressor_test/hp_configs")),
        help="Directory to store results. Results saved under <save_dir>/<job_name>/",
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default=None,
        help="Optional job_name; defaults to the last component of data_dir",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S"
    )

    df = load_latest_csv(Path(args.data_dir), args.regressor)
    feat_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feat_cols].to_numpy(np.float32)
    y = df["target_utility"].to_numpy(np.float32)
    rounds = df["round"].to_numpy(np.int32)

    if args.model == "xgboost":
        device, tree_method, predictor = resolve_xgb_device(args.device)
        logging.info("XGBoost device: %s (%s/%s)", device, tree_method, predictor)
        base_gpu = dict(
            tree_method=tree_method,
            predictor=predictor,
            device=device,
            objective="reg:squarederror",
            eval_metric="rmse",
            verbosity=0
        )
        # Expanded search space tuned for ~hours-long runs.
        # Includes regularization and split heuristics.
        grid = {
            "n_estimators":      [200, 400, 600, 800, 1000],
            "learning_rate":     [0.01, 0.02, 0.05, 0.1],
            "max_depth":         [3, 4, 5, 6, 8, 10],
            "subsample":         [0.6, 0.8, 1.0],
            "colsample_bytree":  [0.6, 0.8, 1.0],
            "min_child_weight":  [1, 5, 10],
            "reg_alpha":         [0.0, 0.1, 0.5],
            "reg_lambda":        [1.0, 2.0, 5.0],
            "gamma":             [0.0, 1.0, 5.0],
        }
        cfgs_full = [ {**base_gpu, **g} for g in ParameterGrid(grid) ]
        if args.max_configs is not None and args.max_configs < len(cfgs_full):
            rng = np.random.default_rng(42)
            idx = rng.choice(len(cfgs_full), size=args.max_configs, replace=False)
            cfgs = [cfgs_full[i] for i in idx]
        else:
            cfgs = cfgs_full
    else:
        # Expanded MLP grid. We keep solver='adam' due to early stopping support.
        grid = {
            "hidden_layer_sizes": [
                (64,), (128,), (256,),
                (64, 64), (128, 64), (256, 128),
                (128, 128, 64),
            ],
            "activation":         ["relu", "tanh"],
            "learning_rate_init": [1e-2, 3e-3, 1e-3, 3e-4, 1e-4],
            "batch_size":         [32, 64, 128, 256],
            "alpha":              [1e-5, 1e-4, 1e-3, 1e-2],
            "solver":             ["adam"],
        }
        cfgs_full = list(ParameterGrid(grid))
        if args.max_configs is not None and args.max_configs < len(cfgs_full):
            rng = np.random.default_rng(42)
            idx = rng.choice(len(cfgs_full), size=args.max_configs, replace=False)
            cfgs = [cfgs_full[i] for i in idx]
        else:
            cfgs = cfgs_full

    # Resolve output directory under hp_configs/<job_name>
    job_name = args.job_name or Path(args.data_dir).name
    save_root = Path(args.save_dir) / job_name
    save_root.mkdir(parents=True, exist_ok=True)

    n_cfg = len(cfgs)
    logging.info("Grid size: %d configurations%s", n_cfg,
                 " (sampled)" if args.max_configs else "")

    results: list[tuple[float, dict]] = []
    t0 = perf_counter()

    if args.n_jobs > 1:
        # parallel evaluation of configs
        executor = ProcessPoolExecutor(max_workers=args.n_jobs)
        futures = {
            executor.submit(eval_cfg,
                            args.model, cfg, X, y, rounds, args.cv_splits): cfg
            for cfg in cfgs
        }

        it = as_completed(futures)
        if tqdm:
            it = tqdm(it, total=n_cfg, desc="configs", unit="cfg")

        for fut in it:
            score = fut.result()
            cfg = futures[fut]
            results.append((score, cfg))
    else:
        # sequential (with progress bar)
        iterator = cfgs if not tqdm else tqdm(cfgs, desc="configs", unit="cfg")
        for cfg in iterator:
            score = eval_cfg(args.model, cfg, X, y, rounds, args.cv_splits)
            results.append((score, cfg))

    results.sort(key=lambda t: t[0])
    best_rmse, best_cfg = results[0]
    elapsed = perf_counter() - t0

    print("\n========== BEST ==========")
    print(f"Model       : {args.model}")
    print(f"Regressor   : {args.regressor}")
    print(f"CV splits   : {args.cv_splits}")
    print(f"n_jobs      : {args.n_jobs}")
    print(f"Mean RMSE   : {best_rmse:.5f}")
    print(f"Hyper-params: {best_cfg}")
    print(f"Elapsed     : {elapsed/60:.1f} min")
    print("==========================")

    # Persist ranking to both the dataset folder and hp_configs/<job_name>
    ranking = [{"rmse": s, **c} for s, c in results]
    out1 = Path(args.data_dir) / f"best_{args.model}_{args.regressor}.json"
    out2 = save_root / f"best_{args.model}_{args.regressor}.json"
    for out in (out1, out2):
        with out.open("w") as f:
            json.dump(ranking, f, indent=2)
        logging.info("Full ranking written to %s", out)

    # Also store a minimal snippet with the single best config
    best_snippet = {k: v for k, v in best_cfg.items() if k not in {"tree_method", "predictor", "device", "verbosity", "objective", "eval_metric"}}
    best_meta = {
        "model": args.model,
        "regressor": args.regressor,
        "cv_splits": args.cv_splits,
        "n_jobs": args.n_jobs,
        "rmse": best_rmse,
        "params": best_snippet,
    }
    with (save_root / f"best_{args.model}_{args.regressor}_params.json").open("w") as f:
        json.dump(best_meta, f, indent=2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
