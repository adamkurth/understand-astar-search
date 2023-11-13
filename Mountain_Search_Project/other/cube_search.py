import heapq
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def heuristic(node, goal):
    return np.sqrt((goal[0] - node[0])**2 + (goal[1] - node[1])**2 + (goal[2] - node[2])**2)

def create_3d_grid_graph(x_size, y_size, z_size):
    G = nx.Graph()
    for x in range(x_size):
        for y in range(y_size):
            for z in range(z_size):
                G.add_node((x, y, z))
                if x > 0:
                    G.add_edge((x, y, z), (x - 1, y, z))
                if y > 0:
                    G.add_edge((x, y, z), (x, y - 1, z))
                if z > 0:
                    G.add_edge((x, y, z), (x, y, z - 1))
    return G

def create_3d_custom_grid_graph(x_size, y_size, z_size):
    G = nx.Graph()
    for x in range(x_size):
        for y in range(y_size):
            for z in range(z_size):
                if (x % 2 == 0 and y % 2 == 0) or z == 0 or z == z_size - 1:
                    G.add_node((x, y, z))
                    if x > 0:
                        G.add_edge((x, y, z), (x - 1, y, z))
                    if y > 0:
                        G.add_edge((x, y, z), (x, y - 1, z))
                    if z > 0:
                        G.add_edge((x, y, z), (x, y, z - 1))
    
    # Add an accessible path from start to goal (you can customize this as needed)
    for z in range(z_size - 1):
        G.add_edge((x_size - 1, y_size - 1, z), (x_size - 1, y_size - 1, z + 1))
    
    return G

def a_star_3d(graph, start, goal):
    open_set = [(0 + heuristic(start, goal), 0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    explored_nodes = []

    while open_set:
        _, current_cost, current = heapq.heappop(open_set)
        if current == goal:
            break

        for neighbor in graph.neighbors(current):
            new_cost = current_cost + graph[current][neighbor].get('weight', 1)
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal)
                heapq.heappush(open_set, (priority, new_cost, neighbor))
                came_from[neighbor] = current

        explored_nodes.append(current)

    return came_from, explored_nodes

import time

def visualize_search_process_3d(graph, start, goal, came_from, explored_nodes):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    pos = {node: node for node in graph.nodes()}
    
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from.get(current)
    path.reverse()
    edges = list(zip(path[:-1], path[1:]))
    
    # Define colors for explored and unexplored nodes
    explored_color = 'lightblue'
    unexplored_color = 'lightgray'
    
    for node in graph.nodes():
        if node in explored_nodes:
            ax.scatter(node[0], node[1], node[2], c=explored_color, s=50, marker='o')
        else:
            ax.scatter(node[0], node[1], node[2], c=unexplored_color, s=50, marker='o')
    
    for i, edge in enumerate(graph.edges()):
        if edge in edges:
            ax.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], [edge[0][2], edge[1][2]], c='red', linewidth=2)
        else:
            ax.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], [edge[0][2], edge[1][2]], c='gray', linewidth=1)
        
        # Pause for 0.01 seconds between each step
        plt.draw()
            
    ax.scatter(start[0], start[1], start[2], c='green', s=50, marker='o', label='Start')
    ax.scatter(goal[0], goal[1], goal[2], c='red', s=50, marker='o', label='Goal')
    
    # Add the legend
    ax.legend()
    
    plt.title(f"Shortest path from {start} to {goal}")
    plt.show(block=True)
    
G = create_3d_grid_graph(5, 5, 5)
start, goal = (0, 0, 0), (4, 4, 2) 
came_from, explored_nodes = a_star_3d(G, start, goal)
visualize_search_process_3d(G, start, goal, came_from, explored_nodes)

G = create_3d_custom_grid_graph(7, 5, 4)  
start, goal = (1, 2, 0), (6, 4, 1)
came_from, explored_nodes = a_star_3d(G, start, goal)
visualize_search_process_3d(G, start, goal, came_from, explored_nodes)
