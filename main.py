import itertools
import os
import time

import pandas as pd

from src.constants import (
    CV_FIGURE_PATH, CV_FOLD_METRICS_PATH, CV_FOLDS, CV_HISTORY_PATH,
    CV_SUMMARY_PATH, FIGURE_DIR, FINAL_EPOCHS, NN_TIMING_PATH, OPTIMAL,
    PARAM_GRID, POLY_ALPHAS, POLY_DEGREES, POLY_GRID_PATH, POLY_SUMMARY_PATH,
    SEARCH_EPOCHS, SEARCH_PATH, SEED, SWEEP_FOLDS, SWEEPS, TABLE_DIR,
)
from src.functions import (
    load_data, make_split, plot_cv_curves, plot_sweeps, run_cv, run_poly,
    run_search, run_sweeps, summarize_cv,
)


def check_search_results(results):
    """Stop if saved search results do not match the current PARAM_GRID."""
    keys = list(PARAM_GRID.keys())
    expected = {
        tuple(str(v) for v in combo)
        for combo in itertools.product(*PARAM_GRID.values())
    }
    found = {
        tuple(str(row[k]) for k in keys)
        for _, row in results.iterrows()
    }
    if found != expected:
        raise RuntimeError(
            f"{SEARCH_PATH} has {len(found & expected)}/{len(expected)} of the "
            f"configurations in PARAM_GRID (incomplete or from an older grid). "
            f"Delete it and rerun."
        )


def check_optimal(results):
    """Warn if OPTIMAL is not the top-ranked configuration of the search."""
    best = results.sort_values("final_loss_valid").iloc[0]
    mismatched = [k for k in OPTIMAL if str(best[k]) != str(OPTIMAL[k])]
    if mismatched:
        print(f"WARNING: OPTIMAL differs from the rank-1 search result in "
              f"{mismatched}. Update src/constants.py if this is unintended.")


def main():
    X, y = load_data()
    train_idx, valid_idx = make_split(len(X), 0.8, SEED)
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    # Stage 1: hyperparameter search (skipped if results already exist)
    if os.path.exists(SEARCH_PATH):
        results = pd.read_csv(SEARCH_PATH)
        check_search_results(results)
        print("Loaded existing search results.")
    else:
        results = run_search(
            X[train_idx], y[train_idx], X[valid_idx], y[valid_idx],
            PARAM_GRID, n_epochs=SEARCH_EPOCHS, out_path=SEARCH_PATH, seed=SEED,
        )
    print(results.sort_values("final_loss_valid").head(5).to_string())
    check_optimal(results)

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

    # Stage 3: one-at-a-time hyperparameter sweeps (3-fold CV)
    # run_sweeps skips any sweep whose CSV already exists
    run_sweeps(X, y, OPTIMAL, SWEEPS, FINAL_EPOCHS, SWEEP_FOLDS, TABLE_DIR, SEED)
    for name in SWEEPS:
        print(pd.read_csv(f"{TABLE_DIR}/sweep_{name}.csv").to_string())
    plot_sweeps(SWEEPS, TABLE_DIR, FIGURE_DIR)

    # Stage 4: polynomial regression baseline
    if os.path.exists(POLY_SUMMARY_PATH):
        print("Loaded existing polynomial results.")
    else:
        grid, poly_summary = run_poly(X, y, POLY_DEGREES, POLY_ALPHAS, CV_FOLDS, SEED)
        grid.to_csv(POLY_GRID_PATH, index=False)
        poly_summary.to_csv(POLY_SUMMARY_PATH, index=False)
    poly_summary = pd.read_csv(POLY_SUMMARY_PATH)
    print(poly_summary.to_string())

    # Training-time comparison (both timings cover 4 folds)
    if os.path.exists(NN_TIMING_PATH):
        nn_seconds = pd.read_csv(NN_TIMING_PATH)["seconds"].iloc[0]
        poly_seconds = poly_summary["seconds"].iloc[0]
        print(f"Training time, 4 folds: neural network {nn_seconds:.1f} s, "
              f"polynomial {poly_seconds:.2f} s "
              f"({nn_seconds / poly_seconds:.0f}x faster)")


if __name__ == "__main__":
    main()