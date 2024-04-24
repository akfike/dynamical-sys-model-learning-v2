import pandas as pd

# Load the cleaned and scaled dataset
scaled_df = pd.read_csv('scaled_internal_dynamics_data.csv')

# Inspect the first few rows of the dataset
print("First 5 rows of the scaled dataset:")
print(scaled_df.head())

# Decide Input Features
# For simplicity, let's assume we'll use the state values at time t to predict the next states at time t+dt
# You can select specific columns based on domain knowledge or experimentation
input_features = [col for col in scaled_df.columns if col.endswith('(t)')]

print("\nSelected input features:")
print(input_features)
