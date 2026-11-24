import os
import time

import pandas as pd

from src.constants import (
    CV_FIGURE_PATH, CV_FOLD_METRICS_PATH, CV_FOLDS, CV_HISTORY_PATH,
    CV_SUMMARY_PATH, FINAL_EPOCHS, OPTIMAL, PARAM_GRID, POLY_ALPHAS,
    POLY_DEGREES, POLY_GRID_PATH, POLY_SUMMARY_PATH, SEARCH_EPOCHS,
    SEARCH_PATH, SEED, SWEEP_FOLDS, SWEEPS,
)
from src.functions import (
    load_data, make_split, plot_cv_curves, plot_sweeps, run_cv, run_poly,
    run_search, run_sweeps, summarize_cv,
)

NN_TIMING_PATH = "tables/nn_timing.csv"
def main():
    X, y = load_data()
    train_idx, valid_idx = make_split(len(X), 0.8, SEED)
    os.makedirs("tables", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # Stage 1: hyperparameter search (skipped if results already exist)
    if os.path.exists(SEARCH_PATH):
        results = pd.read_csv(SEARCH_PATH)
        n_expected = 1
        for values in PARAM_GRID.values():
            n_expected *= len(values)
        if len(results) != n_expected:
            raise RuntimeError(
                f"Search incomplete ({len(results)}/{n_expected}). "
                f"Delete {SEARCH_PATH} and rerun."
            )
        print("Loaded existing search results.")
    else:
        results = run_search(
            X[train_idx], y[train_idx], X[valid_idx], y[valid_idx],
            PARAM_GRID, n_epochs=SEARCH_EPOCHS, out_path=SEARCH_PATH, seed=SEED,
        )
    print(results.sort_values("final_loss_valid").head(5).to_string())

    # Stage 2: 4-fold cross-validation of the optimal network
    if os.path.exists(CV_HISTORY_PATH) and os.path.exists(CV_SUMMARY_PATH):
        history = pd.read_csv(CV_HISTORY_PATH)
        summary = pd.read_csv(CV_SUMMARY_PATH)
        print("Loaded existing CV results.")
    else:
        start = time.time()
        history, fold_metrics = run_cv(X, y, OPTIMAL, FINAL_EPOCHS, CV_FOLDS, SEED)
        nn_seconds = time.time() - start
        summary = summarize_cv(fold_metrics)
        history.to_csv(CV_HISTORY_PATH, index=False)
        fold_metrics.to_csv(CV_FOLD_METRICS_PATH, index=False)
        summary.to_csv(CV_SUMMARY_PATH, index=False)
        pd.DataFrame([{"model": "neural_network", "folds": CV_FOLDS,
                       "epochs": FINAL_EPOCHS, "seconds": nn_seconds}]
                     ).to_csv(NN_TIMING_PATH, index=False)
    plot_cv_curves(history, CV_FIGURE_PATH)
    print(summary.to_string())
    if os.path.exists(NN_TIMING_PATH):
        print(pd.read_csv(NN_TIMING_PATH).to_string())

    # Stage 3: one-at-a-time hyperparameter sweeps (3-fold CV)
    run_sweeps(X, y, OPTIMAL, SWEEPS, FINAL_EPOCHS, SWEEP_FOLDS, "tables", SEED)
    for name in SWEEPS:
        print(pd.read_csv(f"tables/sweep_{name}.csv").to_string())
    plot_sweeps(SWEEPS, "tables", "figures")

    # Stage 4: polynomial regression baseline
    if os.path.exists(POLY_SUMMARY_PATH):
        print("Loaded existing polynomial results.")
    else:
        grid, poly_summary = run_poly(X, y, POLY_DEGREES, POLY_ALPHAS, CV_FOLDS, SEED)
        grid.to_csv(POLY_GRID_PATH, index=False)
        poly_summary.to_csv(POLY_SUMMARY_PATH, index=False)
    print(pd.read_csv(POLY_SUMMARY_PATH).to_string())

if __name__ == "__main__":
    main()