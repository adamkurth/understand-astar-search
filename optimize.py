import mountain_search as ms
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


import random
import time

def collect_data(elevation_func, traffic_func, num_runs, num_nodes, num_connections, x_range, y_range):
    data = []
    for _ in range(num_runs): 
        mountain = ms.MountainPassage(
            elevation_funct=elevation_func, 
            traffic_func=traffic_func,
            num_nodes=num_nodes, 
            num_connections=num_connections,
            x_range=x_range,
            y_range=y_range,
            )
        
    for num_nodes in num_nodes_list:
        # Run A* algorithm and get the convergence rate
        mountain = ms.MountainPassage()
        mountain.generate_nodes()
        convergence_rate = mountain.main(num_nodes)
        
        # Append the values to the dataframe
        df = df.append({"num_nodes": num_nodes, "convergence_rate": convergence_rate}, ignore_index=True)
    return df


def elevation_func_5(x, y):
    return x**2 -3*x*(y**2)


def main():

    mountain_search = ms.MountainSearch()
    mountain_search.main() 
    # Create dataframe
    df = create_df([5, 10, 15, 20, 25, 30, 40, 50])