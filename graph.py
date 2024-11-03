# graph.py

import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from rdkit import Chem


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


dataset_path = 'ChCh-Miner_durgbank-chem.tsv'  # Corrected filename
interactions_df = pd.read_csv(dataset_path, sep='\t')


required_columns = ['Drug1_SMILES', 'Drug2_SMILES', 'Label']
if not all(column in interactions_df.columns for column in required_columns):
    raise ValueError("Dataset must contain 'Drug1_SMILES', 'Drug2_SMILES', and 'Label' columns.")


data_list1 = []
data_list2 = []
labels = []


for idx, row in interactions_df.iterrows():
    smiles1 = row['Drug1_SMILES']
    smiles2 = row['Drug2_SMILES']
    label = row['Label']

    data1 = mol_to_graph_data_obj(smiles1)
    data2 = mol_to_graph_data_obj(smiles2)

    if data1 is None or data2 is None:
        continue

    data_list1.append(data1)
    data_list2.append(data2)
    labels.append(label)


labels = torch.tensor(labels, dtype=torch.float)  # Use dtype=torch.long for multi-class


train_indices, test_indices = train_test_split(
    range(len(labels)), test_size=0.2, stratify=labels, random_state=42
)

train_data1 = [data_list1[i] for i in train_indices]
train_data2 = [data_list2[i] for i in train_indices]
train_labels = labels[train_indices]

test_data1 = [data_list1[i] for i in test_indices]
test_data2 = [data_list2[i] for i in test_indices]
test_labels = labels[test_indices]


torch.save((train_data1, train_data2, train_labels, test_data1, test_data2, test_labels), 'processed_data.pt')
