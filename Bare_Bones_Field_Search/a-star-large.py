import heapq
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random
import os

class Node:
    def __init__(self, state, parent=None, action=None, cost=0, heuristic=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        self.heuristic = heuristic

    def total_cost(self):
        return self.cost + self.heuristic

    def __lt__(self, other):
        return self.total_cost() < other.total_cost()

# map attributes
map_width = 100
map_height = 100

# Create a map with random obstacles for testing
random.seed(42)  #for reproducibility
obstacles = [(random.randint(0, map_width - 1), random.randint(0, map_height - 1)) for _ in range(1000)]

initial_state = (0, 0)
goal_state = (99, 99)

plt.ion()
fig, ax = plt.subplots()
output_folder = 'search_steps'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

step_counter = 0

def astar_search(initial_state, goal_state, heuristic, actions_fn, cost_fn, map_width, map_height, obstacles, fig, ax):
    global step_counter #for global access
    start_node = Node(initial_state, None, None, 0, heuristic(initial_state, goal_state))
    open_list = [(start_node.total_cost(), start_node)]
    visited_nodes = set()

    while open_list:
        current_node = heapq.heappop(open_list)[1]

        if current_node.state == goal_state:
            return build_path(current_node)

        visited_nodes.add(current_node.state)

        for action in actions_fn(current_node.state):
            new_state = action
            if is_valid_cell(new_state, map_width, map_height, obstacles) and new_state not in visited_nodes:
                new_cost = current_node.cost + cost_fn(current_node.state, new_state)
                new_heuristic = heuristic(new_state, goal_state)
                new_node = Node(new_state, current_node, action, new_cost, new_heuristic)
                heapq.heappush(open_list, (new_node.total_cost(), new_node))
                
        visualize_cells(visited_nodes, obstacles, goal_state, map_width, map_height, fig, ax)
        # will create image for every 50 steps (comment out if not needed)
        # if step_counter % 50 == 0:
        #     plt.savefig(f"{output_folder}/step_{step_counter}.png")
        step_counter += 1
        plt.pause(0.001)

    return None

def build_path(node):
    path = []
    while node:
        if node.action:
            path.insert(0, (node.action, node.state))
        node = node.parent
    return path

def heuristic(state, goal_state):
    x1, y1 = state
    x2, y2 = goal_state
    return abs(x1 - x2) + abs(y1 - y2)

def custom_actions(state):
    x, y = state
    possible_actions = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]  # Right, Left, Down, Up
    return [s for s in possible_actions if is_valid_cell(s, map_width, map_height, obstacles)]

def custom_cost_fn(state, new_state):
    return 1  # Cost is 1 for each step

def is_valid_cell(state, map_width, map_height, obstacles):
    x, y = state
    return 0 <= x < map_width and 0 <= y < map_height and state not in obstacles

def visualize_cells(visited_cells, obstacles, goal_state, map_width, map_height, fig, ax):
    grid = [['white' if (x, y) not in obstacles else 'blue' for x in range(map_width)] for y in range(map_height)]
    grid[goal_state[1]][goal_state[0]] = 'green'  # Mark the goal as green
    
    for cell in visited_cells:
        x, y = cell
        if grid[y][x] != 'green':
            grid[y][x] = 'Accent'
    
    ax.clear()
    ax.matshow([[1 if c == 'white' else 0 for c in row] for row in grid], cmap='Accent')
    
    ax.scatter(initial_state[0], initial_state[1], c='red', label='Start')
    ax.scatter(goal_state[0], goal_state[1], c='red', label='Goal')
    ax.legend()

visualize_cells([], obstacles, goal_state, map_width, map_height, fig, ax)
path = astar_search(initial_state, goal_state, heuristic, custom_actions, custom_cost_fn, map_width, map_height, obstacles, fig, ax)

if path:
    print("Path found:")
    for action, state in path:
        print(f"Action: {action}, State: {state}")
    plt.savefig(f"{output_folder}/final_path.png")
#keep the final plot visible
plt.ioff()
plt.show()
