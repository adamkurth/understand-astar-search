import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import os
import model as lm

########## Data Loading ##########

def load_data(file):
    print(os.getcwd())
    # filepath = os.path.join(os.getcwd(), 'convergence_data', file)
        
    data = pd.read_csv(file)
    convergence_time = data['Convergence Time']
    raw_cost = data['Raw Cost']
    total_cost = data['Total Cost']
    traffic_cost = data['Traffic Cost']
    path = data['Path']
    num_nodes = data['Number of Nodes']
    num_connections = data['Number of Connections']
    elevation_func = data['Elevation_Function']
    return data, convergence_time, raw_cost, total_cost, traffic_cost, path, num_nodes, num_connections, elevation_func


def same_data():
    return load_data("all_results_df_same.csv")
    
# def half_data():
#     return load_data("all_results_df_half.csv")
    
# def double_data():
#     return load_data("all_results_df_same.csv")
    


########## Data Anysis ##########

def visualize(data):
    lm.plot('Number of Nodes', 'Convergence Time', data, x_label='Number of Nodes', y_label='Convergence Time', title='Convergence Time vs Number of Nodes')
    # lm.plot('Number of Nodes', 'Raw Cost', data, x_label='Number of Nodes', y_label='Raw Cost', title='Raw Cost vs Number of Nodes')
    # lm.plot('Number of Nodes', 'Total Cost', data, x_label='Number of Nodes', y_label='Total Cost', title='Total Cost vs Number of Nodes')
    # lm.plot('Number of Nodes', 'Traffic Cost', data, x_label='Number of Nodes', y_label='Traffic Cost', title='Traffic Cost vs Number of Nodes')
    # lm.plot('Number of Nodes', 'Number of Connections', data, x_label='Number of Nodes', y_label='Number of Connections', title='Number of Connections vs Number of Nodes')
    
    # lm.plot('Number of Connections', 'Convergence Time', data, x_label='Number of Connections', y_label='Convergence Time', title='Convergence Time vs Number of Connections')
    # lm.plot('Number of Connections', 'Raw Cost', data, x_label='Number of Connections', y_label='Raw Cost', title='Raw Cost vs Number of Connections')
    # lm.plot('Number of Connections', 'Total Cost', data, x_label='Number of Connections', y_label='Total Cost', title='Total Cost vs Number of Connections')
    

def model(data):
    # lm.linear_model(['Number of Nodes', 'Number of Connections'], 'Convergence Time', data, data)

    lm.polynomial_model(['Number of Nodes', 'Number of Connections'], 'Convergence Time', data, degree=2)
    return None

def main():    
    data, convergence_time, raw_cost, total_cost, traffic_cost, path, num_nodes, num_connections, elevation_func = same_data()
    # half_data = half_data()
    # double_data = double_data()
    # visualize(data)
    model(data)


if __name__ == '__main__':
    main()

# Example: all_results_df.groupby('Elevation_Function')['Convergence Time'].mean() will give the average convergence time for each elevation function.