import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt



# Load the graph
G = nx.read_graphml('50000ID_11314OOD.graphml')

for node_id, node_data in G.nodes(data=True):
    if 'embedding' in node_data:
        # Convert string back to numpy array
        #print(node_data['type'], ":", len(node_data['embedding']))
        node_data['embedding'] = np.fromstring(node_data['embedding'], sep=',')

# # To get the degree of all nodes, you can iterate over the nodes
# for node in G.nodes():
#     print(f"The degree of node {node} is: {G.degree(node)}")

# Find isolated nodes
isolated_nodes = list(nx.isolates(G))

# Print the number of isolated nodes
print(f"The number of isolated nodes is: {len(isolated_nodes)}")
isolated_ood_nodes = [node for node in isolated_nodes if node[0] == "O"]
print(isolated_ood_nodes)


# Add edge on isolated nodes 
isolate_node = "isolate_node"
embedding = np.load('isolate_node.npy')
G.add_node(isolate_node, type='topic', bipartite=0, embedding=embedding) 
edges = [(isolate_node, node, 1) for node in isolated_ood_nodes]
G.add_weighted_edges_from(edges)

# Add label -1 for topic nodes and isolate one
for node, data in G.nodes(data=True):
    if 'label' not in data:
        data['label'] = -1


label_to_check = -1

# Find nodes with the specific label and check their degrees
nodes_with_label = [node for node, data in G.nodes(data=True) if data.get('label') == label_to_check]

# Get the degrees of these nodes
degrees = {node: G.degree(node) for node in nodes_with_label}

print(f"The degrees of nodes with label '{label_to_check}' are: {degrees}")

# Define the labels to check
label_to_check = -1
target_label = 1

# Find nodes with the specific label
nodes_with_label = [node for node, data in G.nodes(data=True) if data.get('label') == label_to_check]

# Count the edges to nodes with the target label
total_edge = 0
edge_counts = {}
for node in nodes_with_label:
    count = 0
    for neighbor in G.neighbors(node):
        if G.nodes[neighbor].get('label') == target_label:
            count += 1
            total_edge +=1 
    edge_counts[node] = count

print(f"The number of edges between nodes with label '{label_to_check}' and nodes with label '{target_label}' are: {edge_counts}")
print(total_edge)
# Assuming node labels are stored in node attribute 'label' where 0 represents ID and 1 represents OOD
ID_nodes = [node for node, data in G.nodes(data=True) if data['label'] == 0]
OOD_nodes = [node for node, data in G.nodes(data=True) if data['label'] == 1]
topic_nodes = [node for node, data in G.nodes(data=True) if data['label'] == -1]

# Randomly sample 2500 ID nodes and 500 OOD nodes
#np.random.seed(42)  # For reproducibility
rng = np.random.default_rng()
selected_ID_nodes = rng.choice(ID_nodes, 5000, replace=False)
selected_OOD_nodes = rng.choice(OOD_nodes, 1000, replace=False)

# Combine the selected nodes
selected_nodes = np.concatenate((selected_ID_nodes, selected_OOD_nodes, topic_nodes))

# Create a subgraph with the selected nodes
subG = G.subgraph(selected_nodes)

# Prepare node features and labels, and filter nodes
node_features = []
node_labels = []
node_index = []  # to keep track of node indices that are not 'topic'
train_mask = []
node_mask = []
test_mask = []
ood_counter = 0
id_counter = 0
for i, (node, data) in enumerate(subG.nodes(data=True)):
    node_features.append(data['embedding'])  # Assume 'embedding' is already a list of floats
    #print(data['type'], ":", data['embedding'].shape)

    if data['type'] != 'topic':
        node_mask.append(True)
        if data['label'] == 0 and id_counter <= 500:
            train_mask.append(True)
            id_counter +=1
        elif data['label'] == 1 and ood_counter <= 100:
            train_mask.append(True)
            ood_counter +=1
        else:
            train_mask.append(False) 
        node_labels.append(data['label'])  # Assume 'labels' is an integer 0 or 1
  
    else:
        train_mask.append(False)
        node_mask.append(False)
        node_labels.append(3)  # Assume 'labels' is an integer 0 or 1

    node_index.append(i)

# Convert lists to tensors
# Collect all unique node identifiers
unique_nodes = set(sum(subG.edges(), ()))  # Flatten the list of tuples and convert to a set
# Map each unique node to an index
node_to_index = {node: idx for idx, node in enumerate(unique_nodes)}
# Convert edges to indices
edge_indices = [[node_to_index[src], node_to_index[dst]] for src, dst in subG.edges()]

#print(np.array(node_features))
# Convert the list of index pairs into a PyTorch tensor
edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
x = torch.tensor(np.array(node_features), dtype=torch.float)
y = torch.tensor(node_labels, dtype=torch.float)

# Create the PyTorch Geometric data object
data = Data(x=x, edge_index=edge_index, y=y)


# Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(data.num_node_features, 16)
        self.conv2 = GCNConv(16, out_channels)
        self.linear1 = torch.nn.Linear(16, 8)
        self.linear2 = torch.nn.Linear(8, out_channels)
        self.dropout = torch.nn.Dropout(p=0.3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.sigmoid(self.conv2(x, edge_index))
        # x = self.dropout(x)
        # x = F.relu(self.linear1(x))
        # x = self.dropout(x)
        # x = F.sigmoid(self.linear2(x))
        return x

# Initialize the model
model = GCN(out_channels=1)  # Assuming 2 classes for labels

# Loss Function
criterion = torch.nn.BCELoss()

# Optimizer (using Adam here, but you can use others like SGD)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
def train():
    model.train()  # Set the model to training mode
    optimizer.zero_grad()  # Clear gradients to ready for a new optimization step
    out = model(data)  # Forward pass: compute the predicted outputs
    loss = criterion(out[train_mask].squeeze(), data.y[train_mask])  # Compute the loss only on the masked nodes
    loss.backward()  # Backpropagation: compute gradient of the loss with respect to model parameters
    optimizer.step()  # Adjust model weights based on calculated gradients
    return loss

# Example training for a number of epochs
epochs = 300
for epoch in range(epochs):
    loss = train()
    print(f'Epoch {epoch+1}: Loss: {loss.item()}')



# Convert lists to tensors
# Collect all unique node identifiers
# unique_nodes = set(sum(G.edges(), ()))  # Flatten the list of tuples and convert to a set
# # Map each unique node to an index
# node_to_index = {node: idx for idx, node in enumerate(unique_nodes)}
# # Convert edges to indices
# edge_indices = [[node_to_index[src], node_to_index[dst]] for src, dst in G.edges()]

print("Finish")
node_features = []
node_labels = []
node_index = []  # to keep track of node indices that are not 'topic'
train_mask = []
node_mask = []
test_mask = []
isolated_mask = []
ood_counter = 0
id_counter = 0
for i, (node, data) in enumerate(G.nodes(data=True)):
    node_features.append(data['embedding'])  # Assume 'embedding' is already a list of floats
    #print(data['type'], ":", data['embedding'].shape)

    if data['type'] != 'topic':
        node_mask.append(True)
        node_labels.append(data['label'])  # Assume 'labels' is an integer 0 or 1
  
    else:
        train_mask.append(False)
        node_mask.append(False)
        node_labels.append(3)  # Assume 'labels' is an integer 0 or 1

    if node in isolated_nodes:
        isolated_mask.append(True)
    else:
        isolated_mask.append(False)

    node_index.append(i)
# Convert the list of index pairs into a PyTorch tensor
edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
x = torch.tensor(np.array(node_features), dtype=torch.float)
y = torch.tensor(node_labels, dtype=torch.float)

# Create the PyTorch Geometric data object
data_big = Data(x=x, edge_index=edge_index, y=y)

# y_true = data.y[node_mask].squeeze()
# y_pred = (model(data)[node_mask].squeeze() > 0.5) + 0
y_true = data_big.y[node_mask].squeeze()
y_pred = (model(data_big)[node_mask].squeeze() > 0.5) + 0

#print(data.y.squeeze())
#print(model(data).squeeze())
confusion_matrix = confusion_matrix(y_true, y_pred)
formatted_matrix = np.vectorize(lambda x: int(x))(confusion_matrix)
cm_display = ConfusionMatrixDisplay(confusion_matrix = formatted_matrix, display_labels = ["ID", "OOD"])

# Plot the confusion matrix
fig, ax = plt.subplots()
cm_display.plot(ax=ax, values_format='d')

# Manually set axis labels to avoid scientific notation
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):d}'))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y):d}'))


# Show the plot
plt.show()
# cm_display.plot()
# plt.show() 


# Mapping nodes to confusion matrix results
confusion_details = {
    'true_label': [],
    'pred_label': [],
    'nodes': []
}
isolate_false_counter=0
isolate_true_counter=0
for true, pred, node in zip(y_true, y_pred, isolated_nodes):
    confusion_details['true_label'].append(true)
    confusion_details['pred_label'].append(pred)
    confusion_details['nodes'].append(node)
    if true != pred:
        isolate_false_counter += 1
    else:
        isolate_true_counter +=1

# Print detailed results
for i in range(len(confusion_details['true_label'])):
    print(f"Node: {confusion_details['nodes'][i]}, True Label: {confusion_details['true_label'][i]}, Predicted Label: {confusion_details['pred_label'][i]}")

print("isolate rights: ", isolate_true_counter)
print("isolate wrongs: ", isolate_false_counter)