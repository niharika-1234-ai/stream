import mlflow
from mlflow import MlflowClient
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler


MODEL_NAME = "california-housing-linear-regression"
MODEL_ALIAS = "champion"

EXPERIMENTS = [
    {"scaler": StandardScaler, "test_size": 0.2},
    {"scaler": MinMaxScaler, "test_size": 0.2},
    {"scaler": StandardScaler, "test_size": 0.3},
    {"scaler": MinMaxScaler, "test_size": 0.3},
]


def main():
    mlflow.set_experiment("california-housing-linear-regression")
    housing = fetch_california_housing()
    results = []

    for experiment in EXPERIMENTS:
        scaler_type = experiment["scaler"]
        test_size = experiment["test_size"]
        scaler_name = scaler_type.__name__
        run_name = f"{scaler_name}-test-size-{test_size}"

        features_train, features_test, target_train, target_test = train_test_split(
            housing.data, housing.target, test_size=test_size, random_state=42
        )
        model = make_pipeline(scaler_type(), LinearRegression())
        model.fit(features_train, target_train)
        predictions = model.predict(features_test)
        mae = mean_absolute_error(target_test, predictions)
        r2 = r2_score(target_test, predictions)

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(
                {
                    "scaler": scaler_name,
                    "test_size": test_size,
                    "random_state": 42,
                    "model": "LinearRegression",
                }
            )
            mlflow.log_metrics({"mae": mae, "r2": r2})
            model_info = mlflow.sklearn.log_model(model, name="model")
            run_id = mlflow.active_run().info.run_id

        results.append(
            {
                "run": run_name,
                "run_id": run_id,
                "model_uri": model_info.model_uri,
                "scaler": scaler_name,
                "test_size": test_size,
                "mae": mae,
                "r2": r2,
            }
        )

    print(f"{'Run':<35} {'Test size':>10} {'MAE':>10} {'R2':>10}")
    for result in results:
        print(
            f"{result['run']:<35} "
            f"{result['test_size']:>10.2f} "
            f"{result['mae']:>10.4f} "
            f"{result['r2']:>10.4f}"
        )
    best_run = max(results, key=lambda result: (result["r2"], -result["mae"]))
    best_run = max(results, key=lambda result: (result["r2"], -result["mae"]))
    registered_model = mlflow.register_model(best_run["model_uri"], MODEL_NAME)
    MlflowClient().set_registered_model_alias(
        MODEL_NAME, MODEL_ALIAS, str(registered_model.version)
    )

    print(
        f"\nBest run: {best_run['run']} "
        f"(R2={best_run['r2']:.4f}, MAE={best_run['mae']:.4f})"
    )
    print(
        f"Registered as {MODEL_NAME} version {registered_model.version}; "
        f"alias '{MODEL_ALIAS}' points to this version."
    )


if __name__ == "__main__":
    main()
