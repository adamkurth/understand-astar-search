import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import heapq
import pandas as pd
import time
from mpl_toolkits.mplot3d.art3d import Line3DCollection

class MountainPassage:
    def __init__(self, elevation_funct, traffic_func, num_nodes, num_connections, x_range, y_range):
        self.elevation_funct = elevation_funct
        self.traffic_func = traffic_func
        self.num_nodes = num_nodes
        self.num_connections = num_connections
        self.x_range = x_range
        self.y_range = y_range
        self.nodes = self.generate_nodes()
        self.connections = self.generate_connections()
        self.traffic_weight_matrix = self.traffic_func()
    
    def generate_nodes(self): 
        x_nodes = np.random.uniform(self.x_range[0], self.x_range[1], self.num_nodes)
        y_nodes = np.random.uniform(self.y_range[0], self.y_range[1], self.num_nodes)
        z_nodes = self.elevation_funct(x_nodes, y_nodes)
        nodes = np.vstack((x_nodes, y_nodes, z_nodes)).T
        return nodes
    
    def generate_connections(self):
        connections = []
        for i in range(self.num_connections):
            a, b = np.random.choice(self.num_nodes, size=2, replace=False)
            if (a,b) not in connections and (b,a) not in connections:
                connections.append((a,b))
        return connections
    
    def generate_traffic_weight_matrix(self):
        return np.random.rand(self.num_nodes, self.num_nodes)*10
    
    def line_integral(self, a, b): 
        num_samples = 100
        x = np.linspace(a[0], b[0], num_samples)
        y = np.linspace(a[1], b[1], num_samples)
        z = self.elevation_funct(x, y)
        dx = np.diff(x)
        dy = np.diff(y)
        dz = np.diff(z)
        
        dist = np.sum(np.sqrt(dx**2 + dy**2 + dz**2))
        delta_elevation = np.sum(np.abs(dz))
        turns = np.sum(np.abs(np.diff(np.arctan2(dy,dx))))
        crash_potential = turns * delta_elevation
        cost = dist + max(0, delta_elevation) + turns + 2*crash_potential
        return cost

    def find_furthest_nodes(self):
        max_distance = -1
        start, goal = -1, -1
        for i in range(self.num_nodes):
            for j in range(i+1, self.num_nodes):
                distance = self.line_integral(self.nodes[i], self.nodes[j])
                if distance > max_distance:
                    max_distance = distance
                    start, goal = i, j
        return start, goal
    
    def astar(self, start, goal):
        open_set = [(0, start, [])] # priority queue
        closed_set = set() #visted nodes
        min_cost = float('inf') # initialize to minimum cost to be infinity
        best_paths = [] # initialize best paths to be empty
        
        while open_set:
            current_cost, current, path = heapq.heappop(open_set) #pop node with lowest cost
            if current in closed_set:
                continue
            path = path + [current]
            if current == goal:
                #update parameters
                min_cost = current_cost
                best_paths = path
            elif current_cost == min_cost:
                best_paths.append(path)
                continue
            closed_set.add(current)
            for i in range(self.num_nodes):
                if i not in closed_set:
                    # calculates cost considering the line integral (terrain) and traffic weight matrix.
                    cost = self.line_integral(self.nodes[current], self.nodes[i]) + self.traffic_weight_matrix[current, i]
                    # push new cost, node, and path to priority queue
                    heapq.heappush(open_set, (current_cost + cost, i, path))
        return best_paths

    def plot_path_astar(self, path):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.nodes[:, 0], self.nodes[:, 1], self.nodes[:, 2], c='blue', marker='o', s=50, label='Towns')
        for i in range(len(path) - 1):
            a, b = path[i], path[i+1]
            num_samples = 100
            x = np.linspace(self.nodes[a, 0], self.nodes[b, 0], num_samples)
            y = np.linspace(self.nodes[a, 1], self.nodes[b, 1], num_samples)
            z = self.elevation_funct(x, y)
            ax.plot(x, y, z, c='red', label='Optimized Path')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()
    
    ## plot meshgrid, 
    def plot(self):
        x = np.linspace(self.x_range[0], self.x_range[1], 100)
        y = np.linspace(self.y_range[0], self.y_range[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = self.elevation_funct(X, Y)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot_surface(X, Y, Z, color='whitesmoke', alpha=0.6)

        ax.scatter(self.nodes[:, 0], self.nodes[:, 1], self.nodes[:, 2], c='blue', marker='o', s=50, label='Towns')

        cmap = mcolors.LinearSegmentedColormap.from_list("Traffic", ["green", "yellow", "red"])
        norm = plt.Normalize(vmin=np.min(self.traffic_weight_matrix), vmax=np.max(self.traffic_weight_matrix))
        
        for a, b in self.connections:
            num_samples = 100
            x = np.linspace(self.nodes[a, 0], self.nodes[b, 0], num_samples)
            y = np.linspace(self.nodes[a, 1], self.nodes[b, 1], num_samples)
            z = self.elevation_funct(x, y)

            traffic_weight = self.line_integral(self.nodes[a], self.nodes[b])
            color = cmap(norm(traffic_weight))
            
            ax.plot3D(x, y, z, color=color, linestyle='dotted', linewidth=2)

            mid_point_idx = num_samples // 2 
            ax.text(x[mid_point_idx], y[mid_point_idx], z[mid_point_idx], f'{traffic_weight:.2f}', color='red')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Adding a colorbar legend for traffic weights
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(self.traffic_weight_matrix)
        cbar = plt.colorbar(mappable, ax=ax)
        cbar.set_label('Traffic Weight')
        
        ax.legend()
        plt.show()
    
    

    

#############################
########## Getter methods ##########


def elevation_func_1(x, y):

    return np.sin(x) * np.cos(y) + 3

def elevation_func_2(x, y):
    return 0.5*np.sin(0.5*x) * np.cos(0.5*y) + 3

def elevation_func_3(x, y):
    return 1/10*(np.sin(x)*np.cos(y) + np.cos(x*y))

def elevation_func_4(x, y):
    return np.sin(x)*np.cos(y) + (x**2 - y**2)/10 + (np.sin(2*x)*np.cos(2*y))/4

def elevation_func_5(x, y):
    return x**2 -3*x*(y**2)

def traffic_func():
    return np.random.rand(num_nodes, num_nodes) * 10

########## Analyze methods ##########

def analyze_elevation_function(elevation_func, domain, num_nodes_list, num_connections_list_1, num_connections_list_half, num_connections_list_twice):
    x_range, y_range = domain
    convergence_data = []

    for num_nodes in num_nodes_list:
        for num_connections in num_connections_list_1:
            mountain_passage = MountainPassage(elevation_funct=elevation_func, traffic_func=traffic_func, num_nodes=num_nodes, num_connections=num_connections, x_range=x_range, y_range=y_range)
            print(f"Function {elevation_func.__name__}:")

            start_time = time.time()

            # Uncomment these lines to calculate and show A* path
            start, goal = mountain_passage.find_furthest_nodes()
            print("Furthest Nodes: ", start, goal)
            path = mountain_passage.astar(start, goal)
            print("Path:", path)
            # mountain_passage.plot_astar_optimization(start, goal)
            # mountain_passage.plot_path_astar(path)

            convergence_data.append({
                'Function': elevation_func.__name__,
                'Num Nodes': num_nodes,
                'Num Connections': num_connections,
                'Total Time to Convergence': time.time() - start_time
            })

    return pd.DataFrame(convergence_data)

########## main ##########

def main():
    
    num_nodes_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    num_connections_list_1 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    num_connections_list_half = [x // 2 for x in num_connections_list_1]
    num_connections_list_twice = [2 * x for x in num_connections_list_1]

    elevation_functions = [elevation_func_1, elevation_func_2, elevation_func_3, elevation_func_4, elevation_func_5]
    domains = [((-np.pi, np.pi), (-np.pi, np.pi)),
               ((-2, 2), (-2, 2)),
               ((0, np.pi), (0, np.pi))]

    savemycomputer_functon = elevation_functions[:2]
    savemycomputer_domain = domains[:2]
    
    for function, domain in zip(savemycomputer_functon, savemycomputer_domain):
        convergence_data = analyze_elevation_function(function, domain, num_nodes_list, num_connections_list_1, num_connections_list_half, num_connections_list_twice)
        
    # for elevation_func, domain in zip(elevation_functions, domains):
    #     convergence_data = analyze_elevation_function(elevation_func, domain, num_nodes_list, num_connections_list_1, num_connections_list_half, num_connections_list_twice)
    #     print(convergence_data)

if __name__ == "__main__":
    main()