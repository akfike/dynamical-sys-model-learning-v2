import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Load the cleaned and scaled dataset
scaled_df = pd.read_csv('../data/controlled_dynamics_data.csv')

# Define input features (state values at time t)
input_features = [col for col in scaled_df.columns if col.endswith('(t)')]

# Define target features (state values at time t+dt)
target_features = [col for col in scaled_df.columns if col.endswith('(t+dt)')]

# Split the data into input features (X) and target features (y)
X = scaled_df[input_features]
y = scaled_df[target_features]

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the validation set
y_val_pred = model.predict(X_val)

# Evaluate model performance on validation set
val_mse = mean_squared_error(y_val, y_val_pred)
print("Validation Mean Squared Error:", val_mse)

# Save the trained model to a file
model_file_path = '../models/linear_regression_model_controlled_data.pkl'
joblib.dump(model, model_file_path)
print(f"Trained model saved to {model_file_path}")

# Select a sample from the validation set to plot actual vs predicted
sample_index = 0  # You can change this index to visualize different samples
actual_trajectory = y_val.iloc[sample_index]
predicted_trajectory = y_val_pred[sample_index]

# Plotting the actual vs predicted trajectories
plt.figure(figsize=(10, 6))
plt.plot(actual_trajectory, label='Actual Trajectory', marker='o')
plt.plot(predicted_trajectory, label='Predicted Trajectory', marker='x')
plt.title('Actual vs. Predicted Trajectory for Sample Trajectory')
plt.xlabel('State Index')
plt.ylabel('State Value')
plt.legend()
plt.grid(True)
plt.show()
