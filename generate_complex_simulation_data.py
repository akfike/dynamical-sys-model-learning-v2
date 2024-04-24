import numpy as np
from scipy.signal import cont2discrete
import pandas as pd

def simulate_system_internal_dynamics(A, B, C, dt, num_time_steps, initial_state, non_linear=False):
    """
    Simulate the internal dynamics of a complex system from a given initial state, including non-linear transformations.
    
    Parameters:
        A (numpy.ndarray): The linear state transition matrix of the system.
        B (numpy.ndarray): Control input matrix (here used for non-linear state updates).
        C (numpy.ndarray): Observation matrix, which can also contain non-linear transformations.
        dt (float): The time step for the simulation.
        num_time_steps (int): Number of time steps for the simulation.
        initial_state (numpy.ndarray): Initial state of the system.
        non_linear (bool): Flag to apply non-linear transformations to the dynamics.

    Returns:
        numpy.ndarray: Simulated state transitions.
    """
    # Discretize the system dynamics
    Ad, Bd, _, _, _ = cont2discrete((A, B, np.zeros((1, A.shape[0])), np.zeros((1, 1))), dt)
    states = [initial_state]
    x = initial_state
    for _ in range(1, num_time_steps):
        x = Ad @ x
        if non_linear:
            # Introduce non-linearities such as sin and exponential functions
            x = np.sin(x) + np.exp(-x**2) + Bd @ np.random.rand(B.shape[1], 1)
        states.append(x)
    return np.hstack(states)

def generate_data(A, B, C, dt, num_samples, num_time_steps, x_range, non_linear=False):
    dataset = []
    for sim_id in range(num_samples):
        initial_state = np.array([np.random.uniform(low, high) for low, high in x_range]).reshape(-1, 1)
        states = simulate_system_internal_dynamics(A, B, C, dt, num_time_steps, initial_state, non_linear)
        for i in range(num_time_steps - 1):
            record = {'simulation_id': sim_id}
            # Make sure to reshape the matrix multiplication results to always be 2D
            observed_state_t = (C @ states[:, i].reshape(-1, 1)).reshape(-1)
            observed_state_t_plus_dt = (C @ states[:, i + 1].reshape(-1, 1)).reshape(-1)
            record.update({f'state_{j+1}(t)': observed_state_t[j] for j in range(C.shape[0])})
            record.update({f'state_{j+1}(t+dt)': observed_state_t_plus_dt[j] for j in range(C.shape[0])})
            dataset.append(record)
    return pd.DataFrame(dataset)

# Parameters for a more complex system
n = 4  # Increase the number of states
dt = 0.01  # Time step
A = np.random.randn(n, n)  # Random state transition matrix for more complex dynamics
B = np.random.randn(n, 1)  # Control-like matrix used for injecting non-linear state updates
C = np.eye(n) + np.random.rand(n, n) * 0.1  # Observation matrix with slight randomness
num_samples = 100
num_time_steps = 50
x_range = [(-1, 1) for _ in range(n)]  # Range for each state value

df = generate_data(A, B, C, dt, num_samples, num_time_steps, x_range, non_linear=True)
csv_file_path = 'data/complex_internal_dynamics_data.csv'
df.to_csv(csv_file_path, index=False)
print(f'Data saved to {csv_file_path}')

# Generate dataset for testing
test_df = generate_data(A, B, C, dt, num_samples, num_time_steps, x_range, non_linear=True)

# Save test dataset to CSV
test_csv_file_path = 'data/test_complex_internal_dynamics_data.csv'
test_df.to_csv(test_csv_file_path, index=False)

print(f'Test data saved to {test_csv_file_path}')
