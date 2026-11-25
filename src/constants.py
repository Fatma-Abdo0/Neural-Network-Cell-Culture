DATASET_PATH = "datasets/dataset_cell_growth_7factor.csv"

TARGET_COLS = [
    "Specific Growth Rate [d^-1]",
    "Specific Death Rate [d^-1]",
    "Specific Production Rate [pg cell^-1 d^-1]",
]

SEED = 42

PARAM_GRID = {
    "hidden_layers": [[32, 32], [64, 64], [128, 128]],
    "activation": ["relu", "tanh"],
    "learning_rate": [1e-3, 3e-3],
    "batch_size": [64, 256],
    "weight_decay": [0.0, 1e-4, 1e-3],
}

SEARCH_EPOCHS = 100
SEARCH_PATH = "tables/search_results.csv"

OPTIMAL = {
    "hidden_layers": [32, 32],
    "activation": "relu",
    "learning_rate": 3e-3,
    "batch_size": 256,
    "weight_decay": 0.0,
}
FINAL_EPOCHS = 200
CV_FOLDS = 4
CV_HISTORY_PATH = "tables/cv_history.csv"
CV_FOLD_METRICS_PATH = "tables/cv_fold_metrics.csv"
CV_SUMMARY_PATH = "tables/cv_summary.csv"
CV_FIGURE_PATH = "figures/cv_loss_curves.png"
NN_TIMING_PATH = "tables/nn_timing.csv"

TABLE_DIR = "tables"
FIGURE_DIR = "figures"

SWEEPS = {
    "hidden_layers": [[8], [16, 16], [32, 32], [32, 32, 32], [64, 64, 64]],
    "activation": ["sigmoid", "tanh", "relu", "leakyrelu"],
    "learning_rate": [3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
    "batch_size": [64, 128, 256, 512, 1024],
    "weight_decay": [0.0, 1e-5, 1e-4, 1e-3, 1e-2],
}
SWEEP_FOLDS = 3

POLY_DEGREES = [1, 2, 3, 4, 5, 6]
POLY_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
POLY_GRID_PATH = "tables/poly_grid.csv"
POLY_SUMMARY_PATH = "tables/poly_summary.csv"