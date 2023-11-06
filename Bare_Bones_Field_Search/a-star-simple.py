import heapq
import matplotlib.pyplot as plt

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
        # Define comparison for the priority queue
        return self.total_cost() < other.total_cost()

def astar_search(initial_state, goal_state, heuristic, actions, cost_fn):
    open_list = []
    closed_set = set()

    initial_node = Node(initial_state, None, None, 0, heuristic(initial_state, goal_state))
    heapq.heappush(open_list, (initial_node.total_cost(), initial_node))

    while open_list:
        _, current_node = heapq.heappop(open_list)

        if current_node.state == goal_state:
            return build_path(current_node)

        if current_node.state in closed_set:
            continue

        closed_set.add(current_node.state)

        for action in actions:  # Iterate through the list of actions
            new_state = action(current_node.state)
            if is_valid_cell(new_state):
                new_cost = current_node.cost + cost_fn(current_node.state, new_state)
                new_heuristic = heuristic(new_state, goal_state)
                new_node = Node(new_state, current_node, action, new_cost, new_heuristic)
                heapq.heappush(open_list, (new_node.total_cost(), new_node))
    return None

def build_path(node):
    path = []
    while node:
        if node.action:
            path.insert(0, (node.action, node.state))
        node = node.parent
    return path

def heuristic(state, goal_state):
    # Define your heuristic function (e.g., Manhattan distance)
    x1, y1 = state
    x2, y2 = goal_state
    return abs(x1 - x2) + abs(y1 - y2)

actions = [
    lambda state: (state[0], state[1] - 1),  # Move up
    lambda state: (state[0], state[1] + 1),  # Move down
    lambda state: (state[0] - 1, state[1]),  # Move left
    lambda state: (state[0] + 1, state[1])  # Move right
]

# Define the cost function
def cost_fn(state, new_state):
    return 1  # Cost is 1 for each step

def is_valid_cell(state):
    # Implement a function to check if a cell is valid
    x, y = state
    return 0 <= x < map_width and 0 <= y < map_height and not is_obstacle(state)

def is_obstacle(state):
    return state in obstacles

# Visualization function to display the grid and path
def visualize_path(path):
    grid = [['blue' if is_obstacle((x, y)) else 'white' for x in range(map_width)] for y in range(map_height)]
    fig, ax = plt.subplots()
    ax.matshow([[1 if c == 'white' else 0 for c in row] for row in grid], cmap='gray')
    
    plt.scatter(initial_state[0], initial_state[1], c='red', label='Start')
    plt.scatter(goal_state[0], goal_state[1], c='red', label='Goal')

    for action, state in path:
        grid[state[1]][state[0]] = 'blue'
        ax.clear()
        ax.matshow([[1 if c == 'white' else 0 for c in row] for row in grid], cmap='gray')
        plt.scatter(initial_state[0], initial_state[1], c='red', label='Start')
        plt.scatter(goal_state[0], goal_state[1], c='red', label='Goal')
        plt.scatter(state[0], state[1], c='blue')
        plt.legend()
        plt.pause(0.01)  # Add a pause to display each step

    plt.show()


# Define your map attributes
map_width = 10
map_height = 10

# Create a map with obstacles (you can customize this)
obstacles = [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2), (8, 2), (9, 4)]

initial_state = (0, 0)
goal_state = (9, 9)

path = astar_search(initial_state, goal_state, heuristic, actions, cost_fn)

if path:
    print("Path found:")
    for action, state in path:
        print(f"Action: {action}, State: {state}")
    visualize_path(path)
    plt.imshow([[0 if is_obstacle((x, y)) else 1 for x in range(map_width)] for y in range(map_height)], cmap='gray')
    plt.scatter([state[0] for _, state in path], [state[1] for _, state in path], c='red')
    plt.show()
else:
    print("No path found")
