import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset from CSV
df = pd.read_csv('../data/controlled_dynamics_data.csv')

# Automatically detect number of states
# Assuming the pattern 'state_{i}(t)' and 'state_{i}(t+dt)' for each state
state_columns = [col for col in df.columns if col.endswith('(t)')]
num_states = len(state_columns)

# Number of simulations
if 'simulation_id' in df.columns:
    num_simulations = df['simulation_id'].nunique()
else:
    num_simulations = 1  # Assume only one simulation if no id is provided

# Define the time step used during data generation
dt = 0.01

# Loop through each simulation ID
for sim_id in df['simulation_id'].unique():
    sim_data = df[df['simulation_id'] == sim_id]
    max_time_steps = sim_data.shape[0]
    time = dt * np.arange(max_time_steps)
    
    plt.figure(figsize=(12, 8))
    for state_col in state_columns:
        state_index = state_col.split('_')[1]
        plt.plot(time, sim_data[state_col], label=f'State {state_index}(t)')  # Plot current simulation data

        # Optionally plot state_{i}(t+dt) if available
        state_dt_col = f'state_{state_index}(t+dt)'
        if state_dt_col in df.columns:
            plt.plot(time, sim_data[state_dt_col], label=f'State {state_index}(t+dt)', linestyle='--')

    plt.title(f'Evolution of System States Over Time for Simulation {sim_id}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('State Values')
    plt.legend()
    plt.grid(True)
    plt.show()
