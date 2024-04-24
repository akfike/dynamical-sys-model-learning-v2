import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error

# Assuming that the file you want to validate against is the one with different initial sets
# and the file is saved in the same directory or a known path
# Adjust the file path as needed based on where the script is being run from
test_df = pd.read_csv('../data/internal_dynamics_data_with_sets_and_noise.csv')

# Define input features (state values at time t)
input_features = [col for col in test_df.columns if col.endswith('(t)')]

# Define target features (state values at time t+dt)
target_features = [col for col in test_df.columns if col.endswith('(t+dt)')]

# Split the test data into input features (X_test) and target features (y_test)
X_test = test_df[input_features]
y_test = test_df[target_features]

# Load the trained model
# Ensure the model file name and path are correctly specified
# This file name should match where and what you've named your trained model
model = joblib.load('../models/linear_regression_model_with_sets.joblib')

# Predict on the test set
y_test_pred = model.predict(X_test)

# Evaluate model performance on the test set
test_mse = mean_squared_error(y_test, y_test_pred)
print("Test Mean Squared Error:", test_mse)
