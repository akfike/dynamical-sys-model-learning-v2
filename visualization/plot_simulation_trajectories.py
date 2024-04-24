import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset from CSV
df = pd.read_csv('../data/controlled_dynamics_data.csv')

# Determine the number of simulations
num_simulations = df['simulation_id'].nunique()

# Initialize the plot outside the loop
plt.figure(figsize=(8, 8))
plt.title('State 1 vs. State 2 for All Simulations')
plt.xlabel('State 1')
plt.ylabel('State 2')
plt.grid(True)

# Plotting the evolution of State 1 over State 2 for each simulation
for sim_id in range(num_simulations):
    # Filter the DataFrame for the current simulation ID
    sim_data = df[df['simulation_id'] == sim_id]

    # Extract State 1 and State 2 data
    state_1 = sim_data['state_1(t)']
    state_2 = sim_data['state_2(t)']

    # Plot on the same figure
    plt.plot(state_1, state_2, marker='o', linestyle='-', label=f'Simulation {sim_id + 1}')

# Add a legend to differentiate between simulations
plt.legend()

# Show the combined plot
plt.show()
