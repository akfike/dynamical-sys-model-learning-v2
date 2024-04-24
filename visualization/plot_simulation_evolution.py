import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset from CSV
df = pd.read_csv('../data/internal_dynamics_data_with_noise.csv')

# Automatically detect number of states
# Assuming the pattern 'state_{i}(t)' and 'state_{i}(t+dt)' for each state
state_columns = [col for col in df.columns if col.endswith('(t)')]
num_states = len(state_columns)

# Create a time vector assuming dt = 0.01 and the index represents each time step
dt = 0.01
time = dt * df.index

# Plotting the evolution of each state
plt.figure(figsize=(12, 8))
for state_col in state_columns:
    state_index = state_col.split('_')[1]
    plt.plot(time, df[state_col], label=f'State {state_index}')

plt.title('Evolution of System States Over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('State Values')
plt.legend()
plt.grid(True)
plt.show()
