# gnn.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import DataLoader, Batch
from torch.utils.data import Dataset


train_data1, train_data2, train_labels, test_data1, test_data2, test_labels = torch.load('processed_data.pt')


class MoleculeInteractionDataset(Dataset):
    def __init__(self, data_list1, data_list2, labels):
        self.data_list1 = data_list1
        self.data_list2 = data_list2
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data_list1[idx], self.data_list2[idx], self.labels[idx]


train_dataset = MoleculeInteractionDataset(train_data1, train_data2, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = MoleculeInteractionDataset(test_data1, test_data2, test_labels)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GNNEncoder, self).__init__()
        self.conv1 = GCNConv(6, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  # Global pooling
        return x

class InteractionPredictor(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(InteractionPredictor, self).__init__()
        self.encoder = GNNEncoder(hidden_channels)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1),
        )

    def forward(self, data1_batch, data2_batch):
        emb1 = self.encoder(
            data1_batch.x, data1_batch.edge_index, data1_batch.edge_attr, data1_batch.batch
        )
        emb2 = self.encoder(
            data2_batch.x, data2_batch.edge_index, data2_batch.edge_attr, data2_batch.batch
        )
        combined = torch.cat([emb1, emb2], dim=1)
        out = self.classifier(combined)
        return out


model = InteractionPredictor(hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss()

epochs = 10  

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for data1_list, data2_list, label_batch in train_loader:
        optimizer.zero_grad()

        # Batch the data
        data1_batch = Batch.from_data_list(data1_list)
        data2_batch = Batch.from_data_list(data2_list)

        output = model(data1_batch, data2_batch)
        label_batch = label_batch.view(-1, 1)

        loss = criterion(output, label_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')

# Save the trained model
torch.save(model.state_dict(), 'gnn_model.pth')

# Eval
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data1_list, data2_list, label_batch in test_loader:
        data1_batch = Batch.from_data_list(data1_list)
        data2_batch = Batch.from_data_list(data2_list)
        output = model(data1_batch, data2_batch)
        preds = (torch.sigmoid(output) > 0.5).long()
        correct += (preds.view(-1) == label_batch).sum().item()
        total += label_batch.size(0)
accuracy = correct / total
print(f'Test Accuracy: {accuracy:.4f}')
