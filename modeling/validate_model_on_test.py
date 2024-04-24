import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def predict_trajectory(initial_state, model, num_steps, feature_names):
    if isinstance(initial_state, np.ndarray):
        initial_state = pd.DataFrame([initial_state], columns=feature_names)
    trajectory = [initial_state.values.flatten()]
    current_state = initial_state
    for _ in range(num_steps):
        next_state = model.predict(current_state)[0]
        trajectory.append(next_state)
        current_state = pd.DataFrame([next_state], columns=feature_names)
    return np.array(trajectory)

# Load the dataset with initial states and the trajectories
test_df = pd.read_csv('../data/controlled_dynamics_test_data.csv')

# Load the trained model
model = joblib.load('../models/linear_regression_model_controlled_data.pkl')

# Identify feature names used during model training
feature_names = [col for col in test_df.columns if col.endswith('(t)')]

# Process each initial state to predict and validate trajectories
mse_scores = []
for i in range(100):  # Assuming there are exactly 100 simulations
    initial_index = i * 50
    if initial_index >= len(test_df) - 50:
        break
    initial_state = test_df.iloc[initial_index][feature_names].values
    actual_trajectory = np.vstack([initial_state, test_df.iloc[initial_index + 1: initial_index + 50][[col for col in test_df.columns if col.endswith('(t+dt)')]].values])
    predicted_trajectory = predict_trajectory(initial_state, model, 49, feature_names)

    # Calculate MSE for this trajectory and each state
    mse = mean_squared_error(actual_trajectory.flatten(), predicted_trajectory.flatten())
    mse_scores.append(mse)

    # Plotting for each state variable
    plt.figure(figsize=(12, 6))
    for j in range(len(feature_names)):
        plt.subplot(1, len(feature_names), j+1)
        actual = actual_trajectory[:, j]
        predicted = predicted_trajectory[:, j]
        plt.plot(actual, 'o-', label='Actual')
        plt.plot(predicted, 'x-', label='Predicted')
        plt.title(f'State {j+1} - Simulation {i+1}')
        plt.xlabel('Time Step')
        plt.ylabel(f'State {j+1} Value')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.show()

# Calculate the average MSE across all trajectories
average_mse = np.mean(mse_scores)
print("Average Mean Squared Error across all trajectories:", average_mse)
