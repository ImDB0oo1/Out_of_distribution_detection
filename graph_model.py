import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, data, out_channels):
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
        return x

def train(model, data, criterion, optimizer, train_mask):
    model.train()  # Set the model to training mode
    optimizer.zero_grad()  # Clear gradients to ready for a new optimization step
    out = model(data)  # Forward pass: compute the predicted outputs
    loss = criterion(out[train_mask].squeeze(), data.y[train_mask])  # Compute the loss only on the masked nodes
    loss.backward()  # Backpropagation: compute gradient of the loss with respect to model parameters
    optimizer.step()  # Adjust model weights based on calculated gradients
    return loss

def initialize_training(data, out_channels=1, learning_rate=0.01):
    model = GCN(data=data, out_channels=out_channels)  # Initialize the model
    criterion = torch.nn.BCELoss()  # Loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Optimizer
    
    return model, criterion, optimizer

def train_model(model, data, criterion, optimizer, train_mask, epochs=600):
    for epoch in range(epochs):
        loss = train(model, data, criterion, optimizer, train_mask)
        print(f'Epoch {epoch+1}: Loss: {loss.item()}')
