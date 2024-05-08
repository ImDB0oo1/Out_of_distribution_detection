import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the graph
G = nx.read_graphml('5000ID_2500OOD.graphml')

for node_id, node_data in G.nodes(data=True):
    if 'embedding' in node_data:
        # Convert string back to numpy array
        #print(node_data['type'], ":", len(node_data['embedding']))
        node_data['embedding'] = np.fromstring(node_data['embedding'], sep=',')

for node, data in G.nodes(data=True):
    if 'label' not in data:
        data['label'] = -1

# Assuming node labels are stored in node attribute 'label' where 0 represents ID and 1 represents OOD
ID_nodes = [node for node, data in G.nodes(data=True) if data['label'] == 0]
OOD_nodes = [node for node, data in G.nodes(data=True) if data['label'] == 1]
topic_nodes = [node for node, data in G.nodes(data=True) if data['label'] == -1]

# Randomly sample 2500 ID nodes and 500 OOD nodes
np.random.seed(42)  # For reproducibility
selected_ID_nodes = np.random.choice(ID_nodes, 2500, replace=False)
selected_OOD_nodes = np.random.choice(OOD_nodes, 500, replace=False)

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
        if data['label'] == 0 and id_counter <= 499:
            train_mask.append(True)
            id_counter +=1
        elif data['label'] == 1 and ood_counter <= 149:
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
        self.conv2 = GCNConv(16, 8)
        self.conv3 = GCNConv(8, out_channels)
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.sigmoid(self.conv3(x, edge_index))
        return x

# Initialize the model
model = GCN(out_channels=1)  # Assuming 2 classes for labels

# Loss Function
criterion = torch.nn.BCELoss()

# Optimizer (using Adam here, but you can use others like SGD)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
def train():
    model.train()  # Set the model to training mode
    optimizer.zero_grad()  # Clear gradients to ready for a new optimization step
    out = model(data)  # Forward pass: compute the predicted outputs
    loss = criterion(out[train_mask].squeeze(), data.y[train_mask])  # Compute the loss only on the masked nodes
    loss.backward()  # Backpropagation: compute gradient of the loss with respect to model parameters
    optimizer.step()  # Adjust model weights based on calculated gradients
    return loss

# Example training for a number of epochs
epochs = 1000
for epoch in range(epochs):
    loss = train()
    print(f'Epoch {epoch+1}: Loss: {loss.item()}')



# Convert lists to tensors
# Collect all unique node identifiers
unique_nodes = set(sum(G.edges(), ()))  # Flatten the list of tuples and convert to a set
# Map each unique node to an index
node_to_index = {node: idx for idx, node in enumerate(unique_nodes)}
# Convert edges to indices
edge_indices = [[node_to_index[src], node_to_index[dst]] for src, dst in G.edges()]


node_features = []
node_labels = []
node_index = []  # to keep track of node indices that are not 'topic'
train_mask = []
node_mask = []
test_mask = []
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

    node_index.append(i)
#print(np.array(node_features))
# Convert the list of index pairs into a PyTorch tensor
edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
x = torch.tensor(np.array(node_features), dtype=torch.float)
y = torch.tensor(node_labels, dtype=torch.float)

# Create the PyTorch Geometric data object
data_big = Data(x=x, edge_index=edge_index, y=y)

y_true = data_big.y[node_mask].squeeze()
y_pred = (model(data_big)[node_mask].squeeze() > 0.5) + 0
#print(data.y.squeeze())
#print(model(data).squeeze())
confusion_matrix = confusion_matrix(y_true, y_pred)

cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["ID", "OOD"])

cm_display.plot()
plt.show() 
