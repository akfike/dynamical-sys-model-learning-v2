import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset from CSV
df = pd.read_csv('../data/complex_internal_dynamics_data.csv')

# Automatically detect number of states
# Assuming the pattern 'state_{i}(t)' and 'state_{i}(t+dt)' for each state
state_columns = [col for col in df.columns if col.endswith('(t)')]
num_states = len(state_columns)

# Create a time vector assuming dt = 0.01 and each row represents a subsequent time step
# Calculate the total number of time steps from one simulation assuming each state's data is listed sequentially
# and that the simulation_id or similar field can be used to distinguish between different simulations
if 'simulation_id' in df.columns:
    max_time_steps = df[df['simulation_id'] == df['simulation_id'].iloc[0]].shape[0]
else:
    max_time_steps = df.shape[0]  # Assume only one simulation if no id is provided

dt = 0.01
time = dt * np.arange(max_time_steps)

# Plotting the evolution of each state
plt.figure(figsize=(12, 8))
for state_col in state_columns:
    state_index = state_col.split('_')[1]
    plt.plot(time, df[state_col][:max_time_steps], label=f'State {state_index}(t)')  # Plot only the first simulation set or the single set

    # Optionally plot state_{i}(t+dt) if available
    state_dt_col = f'state_{state_index}(t+dt)'
    if state_dt_col in df.columns:
        plt.plot(time, df[state_dt_col][:max_time_steps], label=f'State {state_index}(t+dt)', linestyle='--')

plt.title('Evolution of System States Over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('State Values')
plt.legend()
plt.grid(True)
plt.show()
