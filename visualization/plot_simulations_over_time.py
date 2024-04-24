import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # Ensure numpy is imported

# Load the dataset from CSV
df = pd.read_csv('../data/internal_dynamics_data_with_noise.csv')

# Automatically detect number of states
state_columns = [col for col in df.columns if col.endswith('(t)')]
num_states = len(state_columns)

# Number of simulations
num_simulations = df['simulation_id'].nunique()

# Define the time step used during data generation
dt = 0.01

# Plotting the evolution of each state for each simulation
fig, axes = plt.subplots(num_states, 1, figsize=(12, 8), sharex=True)

# Ensure axes is an array even if num_states is 1
if num_states == 1:
    axes = [axes]

for index, state_col in enumerate(state_columns):
    for sim_id in range(num_simulations):
        # Filter the DataFrame for the current simulation ID
        sim_data = df[df['simulation_id'] == sim_id]
        
        # Recompute time for each simulation using numpy for element-wise multiplication
        time = dt * np.arange(len(sim_data))

        # Plot on the appropriate subplot
        axes[index].plot(time, sim_data[state_col], label=f'Simulation {sim_id + 1}')

    axes[index].set_title(f'Evolution of {state_col} Over Time')
    axes[index].set_ylabel('State Value')
    axes[index].grid(True)

# Set common labels and legend
axes[-1].set_xlabel('Time (seconds)')
handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
plt.tight_layout()
plt.show()
