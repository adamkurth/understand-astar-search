import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import heapq

# Generate surface of mountain passage
x = np.linspace(0, 10, 1000)
y = np.linspace(-5, 5, 1000)
X, Y = np.meshgrid(x, y)
Z = np.sin(X/2) + np.cos(Y/2) + 2  # Simulate a gentle mountain passage

# generate nodes as town along passage put into 2d array
num_nodes = 10
x_nodes = np.linspace(0, 10, num_nodes)
y_nodes = np.linspace(-5, 5, num_nodes)
z_nodes = np.sin(x_nodes/2) + np.cos(y_nodes/2) + 2 # same function as above

nodes = np.vstack((x_nodes, y_nodes, z_nodes)).T
# print(nodes)

# Simulate traffic as weughts on edges
traffic_weights = np.random.rand(num_nodes, num_nodes)*10

# Cost function considering elevation change
def cost(a, b, traffic_weight):
    dist = np.linalg.norm(a - b)
    elevation_change = b[2] - a[2]
    return dist + max(0, elevation_change) + traffic_weight # Add penalty for uphill

# A* algorithm to find the shortest path
def astar(start, goal, ax):
    open_set = [(0, start, [])]
    closed_set = set()
    while open_set:
        current_cost, current, path = heapq.heappop(open_set)
        if current in closed_set:
            continue
        path = path + [current]
        if current == goal:
            return path
        closed_set.add(current)
        for i in [current-1, current+1]:  # Connect to adjacent nodes
            if 0 <= i < num_nodes and i not in closed_set:
                # Visualize the searching process
                ax.plot([nodes[current, 0], nodes[i, 0]],
                        [nodes[current, 1], nodes[i, 1]],
                        [nodes[current, 2], nodes[i, 2]], c='yellow', linestyle='dotted')
                heapq.heappush(open_set, (current_cost + cost(nodes[current], nodes[i], traffic_weights[current, i]), i, path))
    return []

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D mountain passage
ax.plot_surface(X, Y, Z, cmap='terrain', alpha=0.6)

# Plot nodes (towns)
ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c='blue', marker='o', s=50, label='Towns')

# Plot edges of the mountain road path with traffic weights
for i in range(num_nodes - 1):
    ax.plot([nodes[i, 0], nodes[i+1, 0]],
            [nodes[i, 1], nodes[i+1, 1]],
            [nodes[i, 2], nodes[i+1, 2]], c='gray', linestyle='dotted')
    ax.text((nodes[i, 0] + nodes[i+1, 0]) / 2,
            (nodes[i, 1] + nodes[i+1, 1]) / 2,
            (nodes[i, 2] + nodes[i+1, 2]) / 2,
            f'{traffic_weights[i, i+1]:.2f}', color='black')

# Choose start and goal nodes
start, goal = 0, num_nodes - 1

# Find the shortest path using A* algorithm and visualize the process
path = astar(start, goal, ax)

# Plot the shortest path
for i in range(len(path) - 1):
    ax.plot([nodes[path[i], 0], nodes[path[i+1], 0]],
            [nodes[path[i], 1], nodes[path[i+1], 1]],
            [nodes[path[i], 2], nodes[path[i+1], 2]], c='red')

# Highlight start and goal nodes
ax.scatter(*nodes[start], c='green', marker='x', s=100, label='Start')
ax.scatter(*nodes[goal], c='red', marker='x', s=100, label='Goal')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()
