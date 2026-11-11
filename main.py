from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def main():
    housing = fetch_california_housing()
    features_train, features_test, target_train, target_test = train_test_split(
        housing.data, housing.target, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(features_train, target_train)
    predictions = model.predict(features_test)

    print(f"Mean squared error: {mean_squared_error(target_test, predictions):.2f}")
    print(f"R^2 score: {r2_score(target_test, predictions):.2f}")


if __name__ == "__main__":
    main()
