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
        self.traffic_weight_matrix = self.traffic_func(self.num_nodes)
    
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
        closed_set = set()
        visited_nodes = set()
        min_cost = float('inf') # initialize to minimum cost to be infinity
        # for recording
        best_paths = [] # initialize best paths to be empty
        df_records = []
        start_time = time.time()
        visited_paths = {} # initialize visited paths to be empty
        
        while open_set:
            current_cost, current, path = heapq.heappop(open_set) #pop node with lowest cost
            if current in closed_set:
                continue
            path = path + [current]
            visited_nodes.add(current)
            
            if current == goal:
                #update parameters
                min_cost = current_cost
                best_paths = path
                end_time = time.time()
                total_cost = sum(self.traffic_weight_matrix[path[i], path[i + 1]] for i in range(len(path) - 1))
                
                df_records.append({
                    'Convergence Time': end_time - start_time,
                    'Raw Cost': total_cost - self.traffic_weight_matrix[path[-2], path[-1]],
                    'Total Cost': total_cost,
                    'Traffic Cost': self.traffic_weight_matrix[path[-2], path[-1]],
                    'Path': path,
                    'Number of Nodes': self.num_nodes,
                    'Number of Connections': self.num_connections
                })
                visited_paths[tuple(path)] = visited_nodes.copy()
                return df_records, path, visited_nodes, visited_paths
            elif current_cost < min_cost:
                best_paths = [path]
                min_cost = current_cost
        
            closed_set.add(current)
        
            for i in range(self.num_nodes):
                if i not in closed_set and i not in path:
                    # calculates cost considering the line integral (terrain) and traffic weight matrix.
                    cost = self.line_integral(self.nodes[current], self.nodes[i]) + self.traffic_weight_matrix[current, i]
                    # push new cost, node, and path to priority queue
                    heapq.heappush(open_set, (current_cost + cost, i, path))
                    visited_paths[tuple(path + [i])] = visited_nodes.copy()
        return best_paths
    
    def calculate_density():
        pass
    
    

    def main_optimization_method(self, start, goal, gridsize=100, show_plot=True):
        df_records, path, visited_nodes, visited_paths  = self.astar(start, goal)
        if show_plot:
            # self.plot()
            # self.plot_path_astar(path)
            self.visualize_astar(start, goal, gridsize)
            # self.plot_astar_optimization(path, visited_nodes, gridsize)
            
        return df_records

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
    
    ## plot meshgrid surface, nodes, connections, and traffic weights 
    def plot(self, show_plot=True):
        x = np.linspace(self.x_range[0], self.x_range[1], 100)
        y = np.linspace(self.y_range[0], self.y_range[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = self.elevation_funct(X, Y)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        if show_plot:
            ax.plot_surface(X, Y, Z, color='whitesmoke', alpha=0.6)  # Set surface to a single color

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
     
    def visualize_astar(self, start, goal, gridsize=100):
        num_connections = self.num_connections
        num_nodes = self.num_nodes
        # Run the A* algorithm
        df_records, path, visited_nodes, visited_paths = self.astar(start, goal)

        # Set up the 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Generate the landscape for the plot
        x = np.linspace(self.x_range[0], self.x_range[1], gridsize)
        y = np.linspace(self.y_range[0], self.y_range[1], gridsize)
        X, Y = np.meshgrid(x, y)
        Z = self.elevation_funct(X, Y)
        ax.plot_surface(X, Y, Z, color='whitesmoke', alpha=0.6)
        
        # Limit the number of nodes for plotting
        limited_nodes = self.nodes[:num_nodes]
        ax.scatter(limited_nodes[:, 0], limited_nodes[:, 1], limited_nodes[:, 2], c='blue', marker='o', s=50, label='Towns')

        # Limit the number of connections for plotting
        for i, (a, b) in enumerate(self.connections):
            if i >= num_connections:
                break
            line_x, line_y = np.linspace(limited_nodes[a, 0], limited_nodes[b, 0], 100), np.linspace(limited_nodes[a, 1], limited_nodes[b, 1], 100)
            line_z = self.elevation_funct(line_x, line_y)
            ax.plot3D(line_x, line_y, line_z, color='gray', linestyle='dotted', label='Connections')

        # Plot visited nodes and paths
        for visited_path in visited_paths:
            for i in range(len(visited_path) - 1):
                a, b = visited_path[i], visited_path[i+1]
                line_x, line_y = np.linspace(limited_nodes[a, 0], limited_nodes[b, 0], 100), np.linspace(limited_nodes[a, 1], limited_nodes[b, 1], 100)
                line_z = self.elevation_funct(line_x, line_y)
                ax.plot3D(line_x, line_y, line_z, color='green', linestyle='dotted', linewidth=2, label='Explored Nodes/Connections')
                # plt.pause(0.001)

        # Plot the optimized path
        for i in range(len(path) - 1):
            a, b = path[i], path[i+1]
            line_x, line_y = np.linspace(limited_nodes[a, 0], limited_nodes[b, 0], 100), np.linspace(limited_nodes[a, 1], limited_nodes[b, 1], 100)
            line_z = self.elevation_funct(line_x, line_y)
            ax.plot3D(line_x, line_y, line_z, color='red', linestyle='solid', linewidth=2, label='Optimized Path')
            # plt.pause(0.001)
            traffic_weight = self.line_integral(self.nodes[a], self.nodes[b])
            mid_point_idx = 50
            ax.text(line_x[mid_point_idx], line_y[mid_point_idx], line_z[mid_point_idx], f'{traffic_weight:.2f}', color='red')

        # Set axes labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Visualize the explored nodes and traffic weights
        cmap = mcolors.LinearSegmentedColormap.from_list("Traffic", ["green", "yellow", "red"])
        norm = plt.Normalize(vmin=np.min(self.traffic_weight_matrix), vmax=np.max(self.traffic_weight_matrix))
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(self.traffic_weight_matrix)
        cbar = plt.colorbar(mappable, ax=ax)
        cbar.set_label('Traffic Weight')

        # Set custom legend
        custom_legend = [
            plt.Line2D([0], [0], marker='o', color='w', label='Towns', markersize=10, linestyle='None'),
            plt.Line2D([0], [0], color='red', label='Connections', linewidth=2),
            plt.Line2D([0], [0], color='red', label='Optimized Path', linewidth=2),
            plt.Line2D([0], [0], color='green', label='Explored Nodes/Connections', linestyle='dotted', linewidth=2)
        ]
        ax.legend(handles=custom_legend)

        plt.show()
        
################################ new (below) #########################################

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

def traffic_func(num_nodes):
    return np.random.rand(num_nodes, num_nodes) * 10

def main():
    # lists
    num_nodes_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    num_connections_list_1 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    num_connections_list_half = [x // 2 for x in num_connections_list_1]
    num_connections_list_twice = [2 * x for x in num_connections_list_1]

    elevation_functions = [elevation_func_1, elevation_func_2, elevation_func_3, elevation_func_4, elevation_func_5]
    domains = [((-np.pi, np.pi), (-np.pi, np.pi)), ((-2, 2), (-2, 2)),((0, np.pi), (0, np.pi)) , ((0, np.pi), (0, np.pi)), ((-np.pi, np.pi), (-np.pi, np.pi))]

    global show_visualization
    show_visualization_input = input("Do you want to see visualizations? (yes/no): ").strip().lower()
    show_visualization = show_visualization_input == 'yes'

    # Prompt user to choose the elevation function
    while True:
        try:
            # choose the corresponding elevation function and domain for the function
            function_index = int(input(f"Choose an elevation function (0-{len(domains)-1}): "))
            if function_index < 0 or function_index >= len(domains):
                raise ValueError
            break
        except ValueError:
            print("Invalid input. Please enter a valid index.")

    # Select the corresponding domain
    domain = domains[function_index]

    # create dataframe to store results
    all_results_df = pd.DataFrame(columns=['Convergence Time', 'Raw Cost', 'Total Cost', 'Traffic Cost', 'Path', 'Number of Nodes', 'Number of Connections'])
    run_var = True # enable for loop
    
    # ask for option to run same, half, or double the number of nodes simulation since this is very computationally expensive
    while run_var == True:
        option = input("Enter 'same', 'half', or 'double' to run the corresponding simulation: ").strip().lower()
        if option in ['same', 'half', 'double']:
            break
        else:
            print("Invalid option. Please try again.")

    # run the corresponding for loop
    if option == 'same':
        num_connections_to_use = num_connections_list_1
    elif option == 'half':
        num_connections_to_use = num_connections_list_half
    elif option == 'double': # 'double'
        num_connections_to_use = num_connections_list_twice

    function_choice = elevation_functions[function_index]
    if show_visualization:
        print(f'Visualizations of {function_choice.__name__}.\n')
        print(f'For sake of clarity, will plot only 1 demonstration at:\n Number of Nodes {num_nodes_list[3]} \n Number of Connections {num_connections_to_use[3]}\n')
        mountain_passage = MountainPassage(elevation_funct=function_choice, traffic_func=traffic_func, num_nodes=num_nodes_list[3], num_connections=num_connections_to_use[3], x_range=domain[0], y_range=domain[1])
        start, goal = mountain_passage.find_furthest_nodes()
        df_records, _, _, _ = mountain_passage.astar(start, goal)
        mountain_passage.main_optimization_method(start, goal, gridsize=100, show_plot=True)
    else:
        print('No visualizations will be shown.\n')        
        
    for num_nodes in num_nodes_list:
        for num_connections in num_connections_to_use:
            print(f"Running simulation for {function_choice.__name__} with {num_nodes} nodes and {num_connections} connections")
            mountain_passage = MountainPassage(elevation_funct=function_choice, traffic_func=traffic_func, num_nodes=num_nodes, num_connections=num_connections, x_range=domain[0], y_range=domain[1])
            start, goal = mountain_passage.find_furthest_nodes()
            df_records, _, _, _ = mountain_passage.astar(start, goal)
            temp_df = pd.DataFrame(df_records)
            temp_df['Norm'] = np.linalg.norm(np.array(start) - np.array(goal))
            
            # The np.linalg.norm function calculates the Euclidean distance between the start and goal nodes in the temp_df DataFrame. 
            # The units of the distance depend on the units of the x and y coordinates of the nodes. 
            # The norm is the square root of the sum of the squares of the differences between the corresponding coordinates of the two points.
            
            # temp_df['']

            temp_df['Elevation_Function'] = function_choice.__name__
            all_results_df = pd.concat([all_results_df, temp_df], ignore_index=True)
        
        # mountain_passage.main_optimization_method(start, goal, gridsize=100, show_plot=True)

    # Print the DataFrame after all iterations
    print(all_results_df)
    all_results_df.to_csv(f'all_results_df_{option}_{function_choice.__name__}.csv', index=False)
    run_var=False



if __name__ == "__main__":
    main()

 
    


# num_nodes = 30
# num_connections = 20

# x_range = (-np.pi, np.pi)
# y_range = (-np.pi, np.pi)

# mountain_passage_5 = MountainPassage(elevation_funct=elevation_func_5, traffic_func=traffic_func, num_nodes=num_nodes, num_connections=num_connections, x_range=x_range, y_range=y_range)
# print("Elevation Function with Sine and Cosine Interaction:")
# mountain_passage_5.plot(show_plot=True)
# mountain_passage_5.plot(show_plot=False)

# traffic_df = pd.DataFrame(mountain_passage_5.traffic_weight_matrix)
# print(traffic_df)

# """A* Algorithm"""""

# start, goal = mountain_passage_5.find_furthest_nodes()
# print("Furthest Nodes: ", start, goal)
# path = mountain_passage_5.astar(start, goal)
# print("Path:", path)
# # mountain_passage_5.plot_astar_optimization(start, goal)
# mountain_passage_5.plot_path_astar(path)
# df = mountain_passage_5.plot_astar_optimization_new(start, goal, gridsize=100)
# print(df)


















# """Show network with traffic weights computed using line integral"""
# num_nodes = 15
# num_connections = 15

# # x_range = (-np.pi, np.pi)
# # y_range = (-np.pi, np.pi)

# # """Steep Elevation Function"""""
# # mountain_passage_1 = MountainPassage(elevation_funct=elevation_func_1, traffic_func=traffic_func, num_nodes=num_nodes, num_connections=num_connections, x_range=x_range, y_range=y_range)
# # print("Steep Elevation Function:")
# # mountain_passage_1.plot(show_plot=True)
# # mountain_passage_1.plot(show_plot=False)
# # # print(mountain_passage.traffic_weight_matrix)

# # traffic_df = pd.DataFrame(mountain_passage_1.traffic_weight_matrix)
# # print(traffic_df)

# # """A* Algorithm"""""
# # start, goal = mountain_passage_1.find_furthest_nodes()
# # print("Furthest Nodes: ", start, goal)
# # path = mountain_passage_1.astar(start, goal)
# # print("Path:", path)
# # mountain_passage_1.plot_astar_optimization(start, goal)
# # mountain_passage_1.plot_path_astar(path)

# # """Gentle Elevation Function"""""
# # x_range = (-np.pi, np.pi)
# # y_range = (-np.pi, np.pi)

# # mountain_passage_2 = MountainPassage(elevation_funct=elevation_func_2, traffic_func=traffic_func, num_nodes=num_connections, num_connections=num_connections, x_range=x_range, y_range=y_range)
# # print("Gentle Elevation Function:")
# # mountain_passage_2.plot(show_plot=True)
# # mountain_passage_2.plot(show_plot=False)

# # traffic_df = pd.DataFrame(mountain_passage_2.traffic_weight_matrix)
# # print(traffic_df)

# # """A* Algorithm""" 
# # start, goal = mountain_passage_2.find_furthest_nodes()
# # print("Furthest Nodes: ", start, goal)
# # path = mountain_passage_2.astar(start, goal)
# # print("Path:", path)
# # mountain_passage_2.plot_astar_optimization(start, goal)
# # mountain_passage_2.plot_path_astar(path)

# # """Elevation Function 3"""
# # x_range = (-2, 2)
# # y_range = (-2, 2)

# # mountain_passage_3 = MountainPassage(elevation_funct=elevation_func_3, traffic_func=traffic_func, num_nodes=num_nodes, num_connections=num_connections, x_range=x_range, y_range=y_range)
# # print("Elevation Function with Sine and Cosine Interaction:")
# # mountain_passage_3.plot(show_plot=True)
# # mountain_passage_3.plot(show_plot=False)

# # traffic_df = pd.DataFrame(mountain_passage_3.traffic_weight_matrix)
# # print(traffic_df)

# # """A* Algorithm"""""

# # start, goal = mountain_passage_3.find_furthest_nodes()
# # print("Furthest Nodes: ", start, goal)
# # path = mountain_passage_3.astar(start, goal)
# # print("Path:", path)
# # mountain_passage_3.plot_astar_optimization(start, goal)
# # mountain_passage_3.plot_path_astar(path)

# # """Elevation Function 4"""
# # x_range = (0, np.pi)
# # y_range = (0, np.pi)

# # mountain_passage_4 = MountainPassage(elevation_funct=elevation_func_4, traffic_func=traffic_func, num_nodes=num_nodes, num_connections=num_connections, x_range=x_range, y_range=y_range)
# # print("Elevation Function with Sine and Cosine Interaction:")
# # mountain_passage_4.plot(show_plot=True)
# # mountain_passage_4.plot(show_plot=False)

# # traffic_df = pd.DataFrame(mountain_passage_4.traffic_weight_matrix)
# # print(traffic_df)

# # """A* Algorithm"""""

# # start, goal = mountain_passage_4.find_furthest_nodes()
# # print("Furthest Nodes: ", start, goal)
# # path = mountain_passage_4.astar(start, goal)
# # print("Path:", path)
# # mountain_passage_4.plot_astar_optimization(start, goal)
# # mountain_passage_4.plot_path_astar(path)

# # """Elevation Function 4"""

# num_nodes = 30
# num_connections = 20

# x_range = (-np.pi, np.pi)
# y_range = (-np.pi, np.pi)

# mountain_passage_5 = MountainPassage(elevation_funct=elevation_func_5, traffic_func=traffic_func, num_nodes=num_nodes, num_connections=num_connections, x_range=x_range, y_range=y_range)
# print("Elevation Function with Sine and Cosine Interaction:")
# mountain_passage_5.plot(show_plot=True)
# mountain_passage_5.plot(show_plot=False)

# traffic_df = pd.DataFrame(mountain_passage_5.traffic_weight_matrix)
# print(traffic_df)

# """A* Algorithm"""""

# start, goal = mountain_passage_5.find_furthest_nodes()
# print("Furthest Nodes: ", start, goal)
# path = mountain_passage_5.astar(start, goal)
# print("Path:", path)
# # mountain_passage_5.plot_astar_optimization(start, goal)
# mountain_passage_5.plot_path_astar(path)
# df = mountain_passage_5.plot_astar_optimization_new(start, goal, gridsize=100)
# print(df)