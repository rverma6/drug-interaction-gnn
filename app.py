# app.py

import streamlit as st
import torch
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F


def atom_feature_vector(atom):
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetHybridization()),
        atom.GetNumImplicitHs(),
        int(atom.GetIsAromatic()),
    ]

def bond_feature_vector(bond):
    bt = bond.GetBondType()
    return [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
    ]

def mol_to_graph_data_obj(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # Handle invalid SMILES strings


    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(atom_feature_vector(atom))

    # Extract bond information
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append((i, j))
        edge_index.append((j, i))  # Because graphs are undirected
        edge_attr.append(bond_feature_vector(bond))
        edge_attr.append(bond_feature_vector(bond))

    data = Data(
        x=torch.tensor(atom_features, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
    )
    return data


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

# Load the trained model
model = InteractionPredictor(hidden_channels=64)
model.load_state_dict(torch.load('gnn_model.pth', map_location=torch.device('cpu')))
model.eval()

st.title("GNN-Based Molecule Interaction Predictor")

smiles1 = st.text_input("Enter SMILES for Molecule 1", "")
smiles2 = st.text_input("Enter SMILES for Molecule 2", "")

if st.button("Predict Interaction"):
    if smiles1 and smiles2:
        try:
            data1 = mol_to_graph_data_obj(smiles1)
            data2 = mol_to_graph_data_obj(smiles2)
            if data1 is None or data2 is None:
                st.error("Invalid SMILES strings.")
            else:

                data1_batch = Batch.from_data_list([data1])
                data2_batch = Batch.from_data_list([data2])
                with torch.no_grad():
                    output = model(data1_batch, data2_batch)
                    probability = torch.sigmoid(output).item()
                st.write(f"Interaction Probability: {probability:.2f}")

                # Display Molecule Structures
                mol1 = Chem.MolFromSmiles(smiles1)
                mol2 = Chem.MolFromSmiles(smiles2)
                st.subheader("Molecule Structures")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(Draw.MolToImage(mol1), caption="Molecule 1")
                with col2:
                    st.image(Draw.MolToImage(mol2), caption="Molecule 2")

                if probability > 0.5:
                    st.success("Predicted to interact.")
                else:
                    st.warning("Predicted not to interact.")

        except Exception as e:
            st.error(f"Error processing molecules: {e}")
    else:
        st.error("Please enter valid SMILES strings for both molecules.")
