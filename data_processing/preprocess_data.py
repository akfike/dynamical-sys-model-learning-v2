import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('../data/test_internal_dynamics_data_with_noise.csv')

# Step 1: Inspect the Data
print("First 5 rows of the dataset:")
print(df.head())
print("\nSummary statistics:")
print(df.describe())

# Step 2: Data Preparation
# Check for missing values
if df.isnull().sum().sum() == 0:
    print("\nNo missing values in the dataset.")
else:
    print("\nMissing values found:")
    print(df.isnull().sum())

# Assuming no missing values or after handling them,
# Normalize/Scale the data
scaler = StandardScaler()
state_columns = [col for col in df.columns if col.startswith('state_')]
df_scaled = df.copy()
df_scaled[state_columns] = scaler.fit_transform(df[state_columns])

# Save the scaled data to a new CSV file
scaled_csv_file_path = '../data/scaled_test_internal_dynamics_data_with_noise.csv'
df_scaled.to_csv(scaled_csv_file_path, index=False)
print(f"\nScaled data saved to {scaled_csv_file_path}")

print("\nScaled data preview:")
print(df_scaled.head())
