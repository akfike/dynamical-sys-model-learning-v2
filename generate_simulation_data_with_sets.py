import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.signal import cont2discrete
import pandas as pd

def projectVertices(vertices, Ad):
    newVertices = []
    for vertex in vertices:
        newVertex = Ad @ vertex.reshape(-1, 1)  # Apply the system dynamics to compute new vertex position
        newVertices.append(newVertex.flatten())
    return np.array(newVertices)

def simulate_dynamics(initialVertices, A, dt, endTime):
    Ad, _, _, _, _ = cont2discrete((A, np.zeros((A.shape[0], 1)), np.zeros((1, A.shape[0])), np.zeros((1, 1))), dt)
    reachableSets = [initialVertices.copy()]
    for _ in range(endTime):
        reachableSets.append(projectVertices(reachableSets[-1], Ad))
    return reachableSets

def plotReachableSets(reachableSets, endTime, sim_id):
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, endTime + 1))  # Generate a color map
    for timeStep in range(endTime + 1):
        poly = Polygon(reachableSets[timeStep], closed=True, fill=None, edgecolor=colors[timeStep], alpha=0.5)
        plt.gca().add_patch(poly)

    plt.title(f'Reachable Sets Over Time for Simulation {sim_id + 1}')
    plt.xlabel('State x1')
    plt.ylabel('State x2')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# Define system dynamics matrix A and timestep
A = np.array([[-1, 1], [1, -1]])
dt = 0.01
endTime = 20

# Multiple initial conditions
initialSets = [
    np.array([[-0.1, -0.1], [-0.1, 0.1], [0.1, 0.1], [0.1, -0.1]]),
    np.array([[-0.05, -0.05], [-0.05, 0.05], [0.05, 0.05], [0.05, -0.05]]),
    np.array([[-0.15, -0.1], [-0.15, 0.1], [0.15, 0.1], [0.15, -0.1]])
]

# DataFrame to store data
all_data = []

# Run simulations for different initial sets
for sim_id, initialVertices in enumerate(initialSets):
    reachableSets = simulate_dynamics(initialVertices, A, dt, endTime)
    plotReachableSets(reachableSets, endTime, sim_id)  # Plotting each simulation's results
    # Extract data for CSV
    for timeStep in range(endTime):
        for vertex_id, vertex in enumerate(reachableSets[timeStep]):
            record = {
                'simulation_id': sim_id,
                'vertex_id': vertex_id,
                'state_1(t)': vertex[0],
                'state_2(t)': vertex[1],
                'state_1(t+dt)': reachableSets[timeStep + 1][vertex_id][0],
                'state_2(t+dt)': reachableSets[timeStep + 1][vertex_id][1]
            }
            all_data.append(record)

# Convert to DataFrame and save
df = pd.DataFrame(all_data)
df.to_csv('data/internal_dynamics_data_with_sets.csv', index=False)
print('Data saved to simulation_data.csv')
