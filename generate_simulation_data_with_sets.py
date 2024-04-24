import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.signal import cont2discrete

# Function to project the vertices of the reachable set
def projectVertices(vertices, Ad):
    newVertices = []
    for vertex in vertices:
        newVertex = Ad @ vertex.reshape(-1, 1)  # Apply the system dynamics to compute new vertex position
        newVertices.append(newVertex.flatten())
    return np.array(newVertices)

# Function to plot reachable sets over time
def plotReachableSets(initialVertices, A, dt, startTime, endTime):
    Ad, _, _, _, _ = cont2discrete((A, np.zeros((A.shape[0], 1)), np.zeros((1, A.shape[0])), np.zeros((1, 1))), dt)
    plt.figure(figsize=(10, 6))
    initialPolygon = Polygon(initialVertices, closed=True, fill=None, edgecolor='black', alpha=1, label='Initial Set')
    plt.gca().add_patch(initialPolygon)

    reachableSets = [initialVertices.copy()]
    for _ in range(endTime):
        reachableSets.append(projectVertices(reachableSets[-1], Ad))

    for timeStep in range(startTime, endTime + 1):
        poly = Polygon(reachableSets[timeStep], closed=True, fill=None, edgecolor='blue', label=f'Time {timeStep}' if timeStep == endTime else '')
        plt.gca().add_patch(poly)

    plt.title(f'Reachable Sets Over Time from t={startTime} to t={endTime}')
    plt.xlabel('State x1')
    plt.ylabel('State x2')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.xlim([-0.1, 0.1])  # Adjusted based on expected state space expansion
    plt.ylim([-0.1, 0.1])  # Adjusted based on expected state space expansion
    plt.show()

# Define matrix A for the system dynamics
A = np.array([[-1, 1], [1, -1]])
dt = 0.01  # Time step

# Define initial vertices of the polygonal set
initialVertices = np.array([[-0.1, -0.1], [-0.1, 0.1], [0.1, 0.1], [0.1, -0.1]])

# Time bounds for simulation
startTime = 0
endTime = 20

# Run the function to plot reachable sets
plotReachableSets(initialVertices, A, dt, startTime, endTime)
