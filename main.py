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



centrality = []
# Iterate through the phases and add the number of nodes and edges to the respective lists
nodes = []
edges = []
for i in range(1,12):
  nodes.append(G[i].number_of_nodes())
  edges.append(G[i].number_of_edges())
  centrality.append(nx.degree_centrality(G[i]))


# Create a plot of the evolution of the number of nodes and edges over time
plt.plot(nodes, label="Number of Nodes")

plt.xlabel('Phase')
plt.ylabel('Number of Nodes/Edges')
plt.title('Evolution of Network Size')
plt.legend()
plt.show()

#
plt.plot(nodes, label="Number of Nodes")
plt.xlabel('Phase')
plt.ylabel('Number of Nodes')
plt.legend()
plt.show()

plt.plot(edges, label="Number of Edges")
plt.xlabel('Phase')
plt.ylabel('Number of Edges')
plt.legend()
plt.show()


# Create a plot of the sum of centrality
sum_centrality = []
for i in range(1, 12):
    d = pd.DataFrame(nx.eigenvector_centrality(G[i]).items(), columns=['nodes', 'eigenvector_centrality'])
    centrality.append(d)
    sum_centrality.append(d['eigenvector_centrality'].sum())
  
plt.plot(sum_centrality, label="degree of centrality")
plt.xlabel('Phase')
plt.ylabel('sum of eigenvector centrality')
plt.legend()
plt.show()

# Create a plot of the sum of centrality
ave_clusterness = []
for i in range(1, 12):
    c=nx.average_clustering(G[i])
    ave_clusterness.append(c)


plt.plot(ave_clusterness, label="clusterness")
plt.xlabel('Phase')
plt.ylabel('average clusterness')
plt.legend()
plt.show()


for i in range(1,12):
    G_i = G[i]  # get the network for phase i
    pos = nx.drawing.nx_agraph.graphviz_layout(G_i)  # get the node positions using graphviz_layout
    fig, ax = plt.subplots()
    nx.draw(G_i, pos=pos, with_labels=True,ax=ax)  # draw the graph with node labels
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

# Iterate over the networks and get their coarse patterns
top_most_central_nodes=[]
average_degree_of_top_nodes=[]
for i in range(1, 12):
    degree_centrality = nx.degree_centrality(G[i])
    top_nodes = sorted(degree_centrality, key=degree_centrality.get, reverse=True)[:int(len(degree_centrality) * 0.1)]
    subgraph = G[i].subgraph(top_nodes)
    avg_degree = sum(dict(subgraph.degree()).values()) / len(subgraph)
    top_most_central_nodes.append(len(top_nodes))
    average_degree_of_top_nodes.append(round(avg_degree, 2))
    
plt.plot(top_most_central_nodes, label="num of the criminal enterprises")
plt.plot(average_degree_of_top_nodes, label="their average centrality")
plt.xlabel('Phase')
plt.legend()
plt.show()

coarse_pattern = np.array(top_most_central_nodes)*np.array(average_degree_of_top_nodes)

plt.plot(coarse_pattern, label="total centrality of criminal enterprises")
plt.xlabel('Phase')
plt.legend()
plt.show()

hub_scores = {}
auth_scores = {}
for i in range(1, 12):
    G = nx.from_pandas_adjacency(phases[i], create_using=nx.DiGraph)
    hub_scores[i], auth_scores[i] = nx.algorithms.link_analysis.hits(G, max_iter=1000000)


n1_hub = [hub_scores[i]['n1'] for i in range(1, 12)]
n1_auth = [auth_scores[i]['n1'] for i in range(1, 12)]
n3_hub = [hub_scores[i]['n3'] for i in range(1, 12)]
n3_auth = [auth_scores[i]['n3'] for i in range(1, 12)]

plt.plot(range(1, 12), n1_hub, label='n1 hub')
plt.plot(range(1, 12), n1_auth, label='n1 authority')
plt.plot(range(1, 12), n3_hub, label='n3 hub')
plt.plot(range(1, 12), n3_auth, label='n3 authority')
plt.xlabel('Phase')
plt.ylabel('Score')
plt.legend()
plt.show()

phases = {}
G = {}
for i in range(1,12):
    var_name = "phase" + str(i)
    file_name = "https://raw.githubusercontent.com/ragini30/Networks-Homework/main/" + var_name + ".csv"
    phases[i] = pd.read_csv(file_name, index_col=["players"])
    phases[i].columns = "n" + phases[i].columns
    phases[i].index = phases[i].columns
    phases[i][phases[i] > 0] = 1
    
    if i >= 5:
        phases[i].drop(["n1", "n3"], axis=0, inplace=True)
        phases[i].drop(["n1", "n3"], axis=1, inplace=True)
        
    G[i] = nx.from_pandas_adjacency(phases[i])
    G[i].name = var_name

ave_clusterness = []
for i in range(1, 12):
    c=nx.average_clustering(G[i])
    ave_clusterness.append(c)
plt.plot(ave_clusterness, label="clusterness")
plt.xlabel('Phase')
plt.ylabel('average clusterness')
plt.legend()
plt.show()

nodes = []
edges = []
for i in range(1,12):
  nodes.append(G[i].number_of_nodes())
  edges.append(G[i].number_of_edges())
  centrality.append(nx.degree_centrality(G[i]))


# Create a plot of the evolution of the number of nodes and edges over time
plt.plot(nodes, label="Number of Nodes")
plt.plot(edges, label="Number of edges")
plt.xlabel('Phase')
plt.ylabel('Number of Nodes/Edges')
plt.title('Evolution of Network Size')
plt.legend()
plt.show()

for i in range(5,12):
    G[i] = nx.watts_strogatz_graph(len(G[i]), k=4, p=0.3)
nodes = []
edges = []
for i in range(1,12):
  nodes.append(G[i].number_of_nodes())
  edges.append(G[i].number_of_edges())
  centrality.append(nx.degree_centrality(G[i]))


# Create a plot of the evolution of the number of nodes and edges over time
plt.plot(nodes, label="Number of Nodes")
plt.plot(edges, label="Number of edges")
plt.xlabel('Phase')
plt.ylabel('Number of Nodes/Edges')
plt.legend()
plt.show()

ave_clusterness = []
for i in range(1, 12):
    c=nx.average_clustering(G[i])
    ave_clusterness.append(c)
plt.plot(ave_clusterness, label="clusterness")
plt.xlabel('Phase')
plt.ylabel('average clusterness')
plt.legend()
plt.show()

for i in range(5,12):
    G[i] = nx.watts_strogatz_graph(len(G[i]), k=3, p=0.08)
nodes = []
edges = []
for i in range(1,12):
  nodes.append(G[i].number_of_nodes())
  edges.append(G[i].number_of_edges())
  centrality.append(nx.degree_centrality(G[i]))


# Create a plot of the evolution of the number of nodes and edges over time
plt.plot(nodes, label="Number of Nodes")
plt.plot(edges, label="Number of edges")
plt.xlabel('Phase')
plt.ylabel('Number of Nodes/Edges')
plt.legend()
plt.show()

ave_clusterness = []
for i in range(1, 12):
    c=nx.average_clustering(G[i])
    ave_clusterness.append(c)
plt.plot(ave_clusterness, label="clusterness")
plt.xlabel('Phase')
plt.ylabel('average clusterness')
plt.legend()
plt.show()

top_most_central_nodes=[]
average_degree_of_top_nodes=[]
for i in range(1, 12):
    degree_centrality = nx.degree_centrality(G[i])
    top_nodes = sorted(degree_centrality, key=degree_centrality.get, reverse=True)[:int(len(degree_centrality) * 0.1)]
    subgraph = G[i].subgraph(top_nodes)
    avg_degree = sum(dict(subgraph.degree()).values()) / len(subgraph)
    top_most_central_nodes.append(len(top_nodes))
    average_degree_of_top_nodes.append(round(avg_degree, 2))
    
plt.plot(top_most_central_nodes, label="num of the top central nodes")
plt.xlabel('Phase')
plt.legend()
plt.show()

plt.plot(average_degree_of_top_nodes, label="their average centrality")
plt.xlabel('Phase')
plt.legend()
plt.show()