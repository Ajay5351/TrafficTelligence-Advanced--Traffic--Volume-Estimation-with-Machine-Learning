import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

DATA_PATH = "data/training_data.csv"

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at: {path}")
    return pd.read_csv(path)

def main():
    df = load_data(DATA_PATH)

    # Features and target
    X = df[['hour', 'temp', 'holiday', 'weather_main']]
    y = df['traffic_volume']

    numeric_features = ['hour', 'temp']
    categorical_features = ['holiday', 'weather_main']

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("R2 Score:", r2_score(y_test, y_pred))
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

    # Save model
    joblib.dump(model, "traffic_model.pkl")
    print("Model saved as traffic_model.pkl")

if __name__ == "__main__":
    main()
