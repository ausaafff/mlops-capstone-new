import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import mlflow
import mlflow.sklearn


def main():
    df = pd.read_csv("data/housing.csv")
    X = df[["area"]]
    y = df["price"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Training parameters
    test_size = 0.2
    random_state = 42

    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Evaluate
        score = model.score(X_test, y_test)
        print("Model R^2 Score:", score)

        # Log metrics
        mlflow.log_metric("r2_score", score)

        # Save & log model artifact
        joblib.dump(model, "model.pkl")
        mlflow.sklearn.log_model(model, "linear_model")
        mlflow.log_artifact("model.pkl")


if __name__ == "__main__":
    main()