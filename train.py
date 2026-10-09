"""ML Experiment Tracking with MLflow - California Housing price prediction."""
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
EXPERIMENT_NAME = "california-housing-gbr"

CONFIGS = [
    {"run_name": "Run 1", "max_depth": 3, "learning_rate": 0.1},
    {"run_name": "Run 2", "max_depth": 5, "learning_rate": 0.05},
    {"run_name": "Run 3", "max_depth": 7, "learning_rate": 0.01},
]


def load_and_split():
    """Load California Housing and make a reproducible 80/20 split."""
    data = fetch_california_housing()
    return train_test_split(
        data.data, data.target, test_size=0.2, random_state=RANDOM_SEED
    )


def main():
    X_train, X_val, y_train, y_val = load_and_split()
    print(f"Train: {X_train.shape}, Validation: {X_val.shape}")

    mlflow.set_experiment(EXPERIMENT_NAME)
    results = []

    for cfg in CONFIGS:
        with mlflow.start_run(run_name=cfg["run_name"]):
            # Same model every time; only the hyperparameters change.
            model = GradientBoostingRegressor(
                max_depth=cfg["max_depth"],
                learning_rate=cfg["learning_rate"],
                n_estimators=100,
                random_state=RANDOM_SEED,
            )
            model.fit(X_train, y_train)

            preds = model.predict(X_val)
            rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
            mae = float(mean_absolute_error(y_val, preds))
            r2 = float(r2_score(y_val, preds))

            mlflow.log_param("max_depth", cfg["max_depth"])
            mlflow.log_param("learning_rate", cfg["learning_rate"])
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            # Newer MLflow saves sklearn models with skops, which requires explicitly
            # trusting the tree type. The model was trained by this script, so it is safe.
            mlflow.sklearn.log_model(
                model,
                name="model",
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )

            results.append((cfg, rmse, mae, r2))
            print(f"{cfg['run_name']}: depth={cfg['max_depth']} "
                  f"lr={cfg['learning_rate']} RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")

    best = min(results, key=lambda r: r[1])  # lowest RMSE wins
    print(f"\nBest model: {best[0]['run_name']} "
          f"(max_depth={best[0]['max_depth']}, learning_rate={best[0]['learning_rate']}, "
          f"RMSE={best[1]:.4f})")


if __name__ == "__main__":
    main()
