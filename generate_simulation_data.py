import numpy as np
from scipy.signal import cont2discrete
import pandas as pd

def simulate_system_internal_dynamics(A, dt, num_time_steps, initial_state):
    """
    Simulate the internal dynamics of the system from a given initial state.
    
    Parameters:
        A (numpy.ndarray): The state matrix of the system.
        dt (float): The time step for the simulation.
        num_time_steps (int): Number of time steps for the simulation.
        initial_state (numpy.ndarray): Initial state of the system.

    Returns:
        numpy.ndarray: Simulated state transitions.
    """
    Ad, _, _, _, _ = cont2discrete((A, np.zeros((A.shape[0], 1)), np.zeros((1, A.shape[0])), np.zeros((1, 1))), dt)
    states = [initial_state]
    x = initial_state
    for _ in range(1, num_time_steps):
        x = Ad @ x
        states.append(x)
    return np.hstack(states)

def generate_data(A, dt, num_samples, num_time_steps, x_range):
    dataset = []
    for sim_id in range(num_samples):
        initial_state = np.array([np.random.uniform(low, high) for low, high in x_range]).reshape(-1, 1)
        states = simulate_system_internal_dynamics(A, dt, num_time_steps, initial_state)
        for i in range(num_time_steps - 1):
            record = {'simulation_id': sim_id}
            record.update({f'state_{j+1}(t)': states[j, i] for j in range(A.shape[0])})
            record.update({f'state_{j+1}(t+dt)': states[j, i+1] for j in range(A.shape[0])})
            dataset.append(record)
    return pd.DataFrame(dataset)

# Parameters
n = 2  # Number of states
dt = 0.01  # Time step
A = np.array([[-1, 1], [1, -1]])
num_samples = 100  # Number of different initial states
num_time_steps = 50  # Number of time steps per simulation
x_range = [(-1, 1), (-1, 1)]  # Range for initial state values

df = generate_data(A, dt, num_samples, num_time_steps, x_range)
csv_file_path = 'data/internal_dynamics_data.csv'
df.to_csv(csv_file_path, index=False)
print(f'Data saved to {csv_file_path}')

# Generate dataset for testing
test_df = generate_data(A, dt, num_samples, num_time_steps, x_range)

# Save test dataset to CSV
test_csv_file_path = 'data/test_internal_dynamics_data.csv'
test_df.to_csv(test_csv_file_path, index=False)

print(f'Test data saved to {test_csv_file_path}')
