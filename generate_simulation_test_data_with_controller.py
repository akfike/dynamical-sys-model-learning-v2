import numpy as np
import pandas as pd
from scipy.signal import cont2discrete
import control as ctrl

def simulate_system_internal_dynamics_with_controller(A, B, K, dt, num_time_steps, initial_state):
    """
    Simulate the internal dynamics of the system from a given initial state using a feedback controller.
    
    Parameters:
        A (numpy.ndarray): The continuous-time state matrix of the system.
        B (numpy.ndarray): The continuous-time input matrix of the system.
        K (numpy.ndarray): Feedback control matrix.
        dt (float): The time step for the simulation.
        num_time_steps (int): Number of time steps for the simulation.
        initial_state (numpy.ndarray): Initial state of the system.

    Returns:
        numpy.ndarray: Simulated state transitions.
    """
    Ad, Bd, _, _, _ = cont2discrete((A, B, np.zeros((A.shape[0], 1)), np.zeros((1, 1))), dt)
    states = [initial_state.flatten()]
    x = initial_state
    for _ in range(num_time_steps - 1):
        u = -K @ x
        x = Ad @ x + Bd @ u
        states.append(x.flatten())
    return np.array(states).T

def generate_data(A, B, K, dt, num_samples, num_time_steps, x_range):
    dataset = []
    for sim_id in range(num_samples):
        # Generate random initial states within the specified range
        initial_state = np.array([np.random.uniform(low, high) for low, high in x_range]).reshape(-1, 1)
        states = simulate_system_internal_dynamics_with_controller(A, B, K, dt, num_time_steps, initial_state)
        for i in range(num_time_steps - 1):
            record = {'simulation_id': sim_id}
            record.update({f'state_{j+1}(t)': states[j, i] for j in range(A.shape[0])})
            record.update({f'state_{j+1}(t+dt)': states[j, i+1] for j in range(A.shape[0])})
            dataset.append(record)
    return pd.DataFrame(dataset)

# System Parameters
n = 2  # Number of states
dt = 0.01  # Time step
A = np.array([[-1, 1], [1, -1]])
B = np.array([[1], [0]])
Q2 = np.array([[10., 0], [0, 10.]])  # Medium penalty
R = np.array([[0.01]])  # Minimal penalty on control effort

# Discretize the system
Ad, Bd, _, _, _ = cont2discrete((A, B, np.zeros((A.shape[0], 1)), np.zeros((1, 1))), dt)
# Design the controller
K2, S2, E2 = ctrl.dlqr(Ad, Bd, Q2, R)

# Simulation Parameters for test dataset
num_samples_test = 100  # Number of different initial states for the test set
num_time_steps_test = 50  # Number of time steps per simulation for the test set
x_range_test = [(-2, 2), (-2, 2)]  # New range for initial state values to generate the test set

# Generate test dataset
test_df = generate_data(A, B, K2, dt, num_samples_test, num_time_steps_test, x_range_test)
test_csv_file_path = 'data/controlled_dynamics_test_data.csv'
test_df.to_csv(test_csv_file_path, index=False)
print(f'Test data saved to {test_csv_file_path}')
