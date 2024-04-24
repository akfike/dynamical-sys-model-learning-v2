import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Load the data
data = pd.read_csv('../data/internal_dynamics_data_with_sets.csv')

# Assume the structure is ['simulation_id', 'vertex_id', 'state_1(t)', 'state_2(t)', 'state_1(t+dt)', 'state_2(t+dt)']
features = data[['state_1(t)', 'state_2(t)']]
targets = data[['state_1(t+dt)', 'state_2(t+dt)']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the testing set
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Save the model to disk
model_filename = '../models/linear_regression_model_with_sets.joblib'
joblib.dump(model, model_filename)
print(f'Model saved to {model_filename}')

