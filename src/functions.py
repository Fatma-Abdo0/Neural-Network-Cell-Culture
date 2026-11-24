import itertools
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.classes import FullyConnectedNN
from src.constants import DATASET_PATH, TARGET_COLS


def load_data():
    """Return raw inputs X and targets y as NumPy arrays."""
    df = pd.read_csv(DATASET_PATH)
    feature_cols = [c for c in df.columns if c not in TARGET_COLS]
    return df[feature_cols].to_numpy(), df[TARGET_COLS].to_numpy()


def make_split(n_rows, train_fraction, seed):
    """Return shuffled train and validation row indices."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_rows)
    n_train = int(train_fraction * n_rows)
    return idx[:n_train], idx[n_train:]


def train_model(X_train, y_train, X_valid, y_valid, hidden_layers, activation,
                learning_rate, batch_size, weight_decay, n_epochs, seed=42):
    """Train one network. Return (history DataFrame, metrics dict)."""
    torch.manual_seed(seed)

    # Scale using TRAIN statistics only
    X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
    y_mean, y_std = y_train.mean(axis=0), y_train.std(axis=0)

    Xtr = torch.tensor((X_train - X_mean) / X_std, dtype=torch.float32)
    ytr = torch.tensor((y_train - y_mean) / y_std, dtype=torch.float32)
    Xva = torch.tensor((X_valid - X_mean) / X_std, dtype=torch.float32)
    yva = torch.tensor((y_valid - y_mean) / y_std, dtype=torch.float32)

    model = FullyConnectedNN(Xtr.shape[1], ytr.shape[1], hidden_layers, activation)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    history = []
    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), batch_size):
            b = perm[i:i + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(Xtr[b]), ytr[b])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            loss_train = loss_fn(model(Xtr), ytr).item()
            loss_valid = loss_fn(model(Xva), yva).item()
        history.append(
            {"epoch": epoch, "loss_train": loss_train, "loss_valid": loss_valid}
        )

    # Back to real units
    with torch.no_grad():
        pred = model(Xva).numpy() * y_std + y_mean
    rmse = np.sqrt(((pred - y_valid) ** 2).mean(axis=0))
    nrmse = rmse / (y_valid.max(axis=0) - y_valid.min(axis=0))

    return pd.DataFrame(history), {"rmse": rmse, "nrmse": nrmse}


def run_search(X_train, y_train, X_valid, y_valid, param_grid, n_epochs,
               out_path, seed=42):
    """Train every combination in param_grid. Save results to CSV."""
    keys = list(param_grid.keys())
    rows = []
    for values in itertools.product(*param_grid.values()):
        params = dict(zip(keys, values))
        history, metrics = train_model(
            X_train, y_train, X_valid, y_valid,
            n_epochs=n_epochs, seed=seed, **params
        )
        row = {k: str(v) for k, v in params.items()}
        row["best_loss_valid"] = history["loss_valid"].min()
        row["best_epoch"] = int(history["loss_valid"].idxmin()) + 1
        row["final_loss_valid"] = history["loss_valid"].iloc[-1]
        row["mean_nrmse"] = float(metrics["nrmse"].mean())
        rows.append(row)
        print(len(rows), row, flush=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)  # save as we go
    return pd.DataFrame(rows)


def run_cv(X, y, params, n_epochs, n_folds, seed=42):
    """K-fold cross-validation. Return (history, per-fold metrics)."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    histories, rows = [], []
    for fold, (tr, va) in enumerate(kf.split(X), start=1):
        history, metrics = train_model(
            X[tr], y[tr], X[va], y[va], n_epochs=n_epochs, seed=seed, **params
        )
        history["fold"] = fold
        histories.append(history)
        for name, r, n in zip(TARGET_COLS, metrics["rmse"], metrics["nrmse"]):
            rows.append({"fold": fold, "target": name,
                         "rmse": float(r), "nrmse": float(n)})
        print("fold", fold, "done", flush=True)
    return pd.concat(histories, ignore_index=True), pd.DataFrame(rows)


def summarize_cv(fold_metrics):
    """Mean and standard deviation of RMSE and NRMSE for each target."""
    return fold_metrics.groupby("target", sort=False).agg(
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        nrmse_mean=("nrmse", "mean"), nrmse_std=("nrmse", "std"),
    ).reset_index()


def plot_cv_curves(history, out_path):
    """2 x 2 figure: training and validation loss vs epoch for each fold."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    for ax, (fold, h) in zip(axes.ravel(), history.groupby("fold")):
        ax.plot(h["epoch"], h["loss_train"], label="loss_train")
        ax.plot(h["epoch"], h["loss_valid"], label="loss_valid")
        ax.set_title(f"Fold {fold}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE loss (scaled targets)")
        ax.legend()
        ax.set_ylim(0.20, 0.30)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_sweeps(X, y, base, sweeps, n_epochs, n_folds, out_dir, seed=42):
    """Vary one hyperparameter at a time. Save one CSV per sweep."""
    for name, values in sweeps.items():
        path = f"{out_dir}/sweep_{name}.csv"
        if os.path.exists(path):
            print("Skipping", name, "(results exist)")
            continue
        rows = []
        for v in values:
            print("Sweep", name, "=", v, flush=True)
            history, fm = run_cv(X, y, {**base, name: v}, n_epochs, n_folds, seed)
            last = history[history["epoch"] == n_epochs]
            row = {
                name: str(v),
                "mean_nrmse": fm["nrmse"].mean(),
                "nrmse_std_across_folds": fm.groupby("fold")["nrmse"].mean().std(),
                "final_loss_train": last["loss_train"].mean(),
                "final_loss_valid": last["loss_valid"].mean(),
            }
            for target, val in fm.groupby("target", sort=False)["nrmse"].mean().items():
                row["nrmse " + target] = val
            rows.append(row)
        pd.DataFrame(rows).to_csv(path, index=False)


def plot_sweeps(sweeps, table_dir, fig_dir):
    """One figure per sweep: mean NRMSE and final train/valid loss."""
    for name in sweeps:
        df = pd.read_csv(f"{table_dir}/sweep_{name}.csv")
        labels = df[name].astype(str)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        ax1.bar(labels, df["mean_nrmse"], yerr=df["nrmse_std_across_folds"],
                capsize=4)
        ax1.set_xlabel(name)
        ax1.set_ylabel("mean NRMSE (3-fold CV)")
        ax1.set_title(f"Error vs {name}")
        ax2.plot(labels, df["final_loss_train"], marker="o", label="loss_train")
        ax2.plot(labels, df["final_loss_valid"], marker="o", label="loss_valid")
        ax2.set_xlabel(name)
        ax2.set_ylabel("final MSE loss (scaled targets)")
        ax2.set_title(f"Train vs validation loss vs {name}")
        ax2.legend()
        for ax in (ax1, ax2):
            plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        fig.tight_layout()
        fig.savefig(f"{fig_dir}/sweep_{name}.png", dpi=200)
        plt.close(fig)

def poly_cv(X, y, degree, alpha, n_folds, seed=42):
    """K-fold CV of scaled polynomial ridge regression. Return (metrics, seconds)."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rows = []
    start = time.time()
    for fold, (tr, va) in enumerate(kf.split(X), start=1):
        model = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree, include_bias=False),
            Ridge(alpha=alpha),
        )
        model.fit(X[tr], y[tr])
        pred = model.predict(X[va])
        rmse = np.sqrt(((pred - y[va]) ** 2).mean(axis=0))
        nrmse = rmse / (y[va].max(axis=0) - y[va].min(axis=0))
        for name, r, n in zip(TARGET_COLS, rmse, nrmse):
            rows.append({"fold": fold, "target": name,
                         "rmse": float(r), "nrmse": float(n)})
    return pd.DataFrame(rows), time.time() - start


def run_poly(X, y, degrees, alphas, n_folds, seed=42):
    """Grid-search degree and alpha. Return (grid table, best summary table)."""
    grid_rows = []
    best = None
    for degree in degrees:
        for alpha in alphas:
            fm, seconds = poly_cv(X, y, degree, alpha, n_folds, seed)
            mean_nrmse = fm["nrmse"].mean()
            grid_rows.append({"degree": degree, "alpha": alpha,
                              "mean_nrmse": mean_nrmse, "seconds": seconds})
            print("poly", degree, alpha, round(mean_nrmse, 4), flush=True)
            if best is None or mean_nrmse < best[0]:
                best = (mean_nrmse, degree, alpha, fm, seconds)
    summary = summarize_cv(best[3])
    summary["degree"] = best[1]
    summary["alpha"] = best[2]
    summary["seconds"] = best[4]
    return pd.DataFrame(grid_rows), summary