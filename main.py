import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pygraphviz

phases = {}
G = {}
for i in range(1,12): 
  var_name = "phase" + str(i)
  file_name = "https://raw.githubusercontent.com/ragini30/Networks-Homework/main/" + var_name + ".csv"
  phases[i] = pd.read_csv(file_name, index_col = ["players"])
  phases[i].columns = "n" + phases[i].columns
  phases[i].index = phases[i].columns
  phases[i][phases[i] > 0] = 1
  G[i] = nx.from_pandas_adjacency(phases[i])
  G[i].name = var_name

# Create lists to store the number of nodes and edges at each phase
nodes = []
edges = []
centrality = []

# Iterate through the phases and add the number of nodes and edges to the respective lists
for i in range(1,12):
  nodes.append(G[i].number_of_nodes())
  edges.append(G[i].number_of_edges())
  centrality.append(nx.degree_centrality())

# Create a plot of the evolution of the number of nodes and edges over time
plt.plot(nodes, label="Number of Nodes")
plt.plot(edges, label="Number of Edges")
plt.xlabel('Phase')
plt.ylabel('Number of Nodes/Edges')
plt.title('Evolution of Network Size')
plt.legend()
plt.show()

for i in range(1,12):
    G_i = G[i]  # get the network for phase i
    pos = nx.drawing.nx_agraph.graphviz_layout(G_i)  # get the node positions using graphviz_layout
    nx.draw(G_i, pos=pos, with_labels=True)  # draw the graph with node labels
    plt.title(f"Phase {i}")  # set the title of the plot to indicate the phase number
    plt.show() 


# Phase 3
G_3 = G[3]  # get the network for phase 3
dc_3 = nx.degree_centrality(G_3)  # compute the degree centrality for all nodes in G_3
dc_3_norm = {node: round(dc_3[node], 4) for node in ['n1', 'n3', 'n12', 'n83']}  # extract degree centrality for the specified nodes and round to 3 decimal places
print("Degree centrality at phase 3:", dc_3_norm)

# Phase 9
G_9 = G[9]  # get the network for phase 9
dc_9 = nx.degree_centrality(G_9)  # compute the degree centrality for all nodes in G_9
dc_9_norm = {node: round(dc_9[node], 4) for node in ['n1', 'n3', 'n12', 'n83']}  # extract degree centrality for the specified nodes and round to 3 decimal places
print("Degree centrality at phase 9:", dc_9_norm)

# Phase 3
G_3 = G[3]  # get the network for phase 3
dc_3 = nx.betweenness_centrality(G_3,normalized='True')  # compute the degree centrality for all nodes in G_3
dc_3_norm = {node: round(dc_3[node], 4) for node in ['n1', 'n3', 'n12', 'n83']}  # extract degree centrality for the specified nodes and round to 3 decimal places
print("Degree centrality at phase 3:", dc_3_norm)

# Phase 9
G_9 = G[9]  # get the network for phase 9
dc_9 = nx.betweenness_centrality(G_9,normalized='True')  # compute the degree centrality for all nodes in G_9
dc_9_norm = {node: round(dc_9[node], 4) for node in ['n1', 'n3', 'n12', 'n83']}  # extract degree centrality for the specified nodes and round to 3 decimal places
print("Degree centrality at phase 9:", dc_9_norm)

# Phase 3
G_3 = G[3]  # get the network for phase 3
dc_3 = nx.eigenvector_centrality(G_3)  # compute the degree centrality for all nodes in G_3
dc_3_norm = {node: round(dc_3[node], 4) for node in ['n1', 'n3', 'n12', 'n83']}  # extract degree centrality for the specified nodes and round to 3 decimal places
print("Degree centrality at phase 3:", dc_3_norm)

# Phase 9
G_9 = G[9]  # get the network for phase 9
dc_9 = nx.eigenvector_centrality(G_9)  # compute the degree centrality for all nodes in G_9
dc_9_norm = {node: round(dc_9[node], 4) for node in ['n1', 'n3', 'n12', 'n83']}  # extract degree centrality for the specified nodes and round to 3 decimal places
print("Degree centrality at phase 9:", dc_9_norm)

# Between_centrality&eigenvector
nodes_score = pd.DataFrame()
for i in range(1,12):
    G_i = G[i]  # get the network for phase 3
    dc = pd.DataFrame.from_dict(nx.betweenness_centrality(G_i),orient='index')  # compute the degree centrality for all nodes in G_3
    nodes_score = pd.merge(nodes_score,dc,left_index=True, right_index=True,how='outer')
nodes_score = np.array(nodes_score.fillna(0))
nodes_score = np.divide(np.sum(nodes_score, axis=1), np.count_nonzero(nodes_score, axis=1))


nodes_score1 = pd.DataFrame()
for i in range(1,12):
    G_i = G[i]  # get the network for phase 3
    dc = pd.DataFrame.from_dict(nx.betweenness_centrality(G_i),orient='index')  # compute the degree centrality for all nodes in G_3
    nodes_score1 = pd.merge(nodes_score1,dc,left_index=True, right_index=True,how='outer')
nodes_score1 = np.array(nodes_score1.fillna(0))
nodes_score1 = np.divide(np.sum(nodes_score1, axis=1), np.count_nonzero(nodes_score1, axis=1))


dc = pd.DataFrame.from_dict(nx.betweenness_centrality(G[1]),orient='index',columns=['betweenness_centrality'])
print(type(dc))