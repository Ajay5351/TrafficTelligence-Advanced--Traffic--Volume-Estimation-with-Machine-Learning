import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
data = pd.read_csv('data/traffic_data.csv')

# Split features and target
X = data.drop('traffic_volume', axis=1)
y = data['traffic_volume']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'model/traffic_model.pkl')
print("Model trained and saved successfully.")
