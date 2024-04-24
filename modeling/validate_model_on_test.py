import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error

# Load the test dataset
test_df = pd.read_csv('../data/scaled_test_internal_dynamics_data_with_noise.csv')

# Define input features (state values at time t)
input_features = [col for col in test_df.columns if col.endswith('(t)')]

# Define target features (state values at time t+dt)
target_features = [col for col in test_df.columns if col.endswith('(t+dt)')]

# Split the test data into input features (X_test) and target features (y_test)
X_test = test_df[input_features]
y_test = test_df[target_features]

# Load the trained model
model = joblib.load('../models/noisy_linear_regression_model.pkl')

# Predict on the test set
y_test_pred = model.predict(X_test)

# Evaluate model performance on the test set
test_mse = mean_squared_error(y_test, y_test_pred)
print("Test Mean Squared Error:", test_mse)
