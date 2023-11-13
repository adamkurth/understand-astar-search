import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import plot_2 as p

# Load data
def load():
    data = pd.read_csv('data.csv')
    num_nodes = data['Nodes']
    nodes_in_path = data['Number of Nodes in Path']
    cost = data['Cost']
    convergence_time = data['Convergence Time']
    return data,num_nodes,nodes_in_path,cost,convergence_time

def main():
    data, num_nodes, nodes_in_path, cost, convergence_time = load()
    print(data)
    
    p.plot('Nodes', 'Cost', data, x_label='Number of Nodes', y_label='Cost', title='Cost vs Number of Nodes')


if __name__ == '__main__':
    main()
