import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

G = nx.read_graphml('50ID_20OOD.graphml')

for node_id, node_data in G.nodes(data=True):
    if 'embedding' in node_data:
        # Convert string back to numpy array
        node_data['embedding'] = np.fromstring(node_data['embedding'], sep=',')


topics = ["watching", "episode", "movie", "film", "like", "good", "bad", "arts", "think", "horror", "action", "story", "theatrer", "filmmakers", "performance", "name"]

# # Plot graph
# # Get one set of the bipartite graph
# if nx.is_bipartite(G):
#     print("done!")
# nodes_set_1, nodes_set_2 = nx.bipartite.sets(G)

# # Create a position layout: Place nodes from nodes_set_1 at x=0 and nodes_set_2 at x=1
# pos = {node: (0, index) for index, node in enumerate(nodes_set_1)}
# pos.update({node: (1, index) for index, node in enumerate(nodes_set_2)})

# # Draw the graph
# nx.draw(G, pos, with_labels=True, node_color=['skyblue' if node in nodes_set_1 else 'lightgreen' for node in G], node_size=500, font_weight='bold', edge_color='gray')
# plt.show()

plt.figure(figsize=(10,8))
pos = nx.bipartite_layout(G, topics)
#nx.draw_networkx_nodes(G, pos, nodelist=topics, node_color='r', label="topics")
#nx.draw_networkx_nodes(G, pos, nodelist=doc_embeddings, node_color='b', label="document")
nx.draw(G, pos, with_labels=True, node_size=500, font_size=5, font_weight='bold')
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size = 8)
labels = nx.get_node_attributes(G, 'label')
nx.draw_networkx_labels(G, pos, labels=labels)
plt.title('Graph of topics and documents')
plt.show()