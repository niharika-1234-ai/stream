
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


print("Loading California Housing Dataset...")

data = fetch_california_housing(as_frame=True)

df = data.frame

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

print("Dataset Loaded Successfully")
print("Dataset Shape:", df.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)


mlflow.set_experiment("Ridge_Hyperparameter_Preprocessing")


alpha_values = [0.01, 0.1, 1.0, 10.0, 100.0]

preprocessing_methods = [
    "Without Standardization",
    "With Standardization"
]

results = []

print("\nStarting MLflow Experiments...\n")

for preprocessing in preprocessing_methods:

    for alpha in alpha_values:

        with mlflow.start_run():


            if preprocessing == "With Standardization":

                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=alpha))
                ])

            else:

                model = Ridge(alpha=alpha)


            model.fit(X_train, y_train)


            y_pred = model.predict(X_test)


            mse = mean_squared_error(y_test, y_pred)

            rmse = np.sqrt(mse)

            mae = mean_absolute_error(y_test, y_pred)

            r2 = r2_score(y_test, y_pred)


            mlflow.log_param("model", "Ridge Regression")

            mlflow.log_param("alpha", alpha)

            mlflow.log_param(
                "preprocessing",
                preprocessing
            )

            mlflow.log_param("test_size", 0.2)

            mlflow.log_metric("MSE", mse)

            mlflow.log_metric("RMSE", rmse)

            mlflow.log_metric("MAE", mae)

            mlflow.log_metric("R2_Score", r2)

            # Log Model using updated MLflow syntax

            mlflow.sklearn.log_model(
                sk_model=model,
                name="ridge_model"
            )

            # Save Results

            results.append({
                "Preprocessing": preprocessing,
                "Alpha": alpha,
                "MSE": mse,
                "RMSE": rmse,
                "MAE": mae,
                "R2 Score": r2
            })

            print("-----------------------------------")
            print("Preprocessing:", preprocessing)
            print("Alpha:", alpha)
            print("MSE:", round(mse, 4))
            print("RMSE:", round(rmse, 4))
            print("MAE:", round(mae, 4))
            print("R2 Score:", round(r2, 4))
            print("-----------------------------------")



results_df = pd.DataFrame(results)

print("\n\n========== EXPERIMENT RESULTS ==========")

print(results_df.to_string(index=False))


best_result = results_df.loc[
    results_df["RMSE"].idxmin()
]

print("\n========== BEST MODEL ==========")

print("Best Preprocessing:", best_result["Preprocessing"])

print("Best Alpha:", best_result["Alpha"])

print("Best RMSE:", round(best_result["RMSE"], 4))

print("Best R2 Score:", round(best_result["R2 Score"], 4))


results_df.to_csv(
    "ridge_experiment_results.csv",
    index=False
)

print("\nResults saved to ridge_experiment_results.csv")

print("\nAll MLflow Experiments Completed Successfully!")
