# GNN-Based Molecule Interaction Predictor

A Graph Neural Network (GNN) application to predict molecular properties and interactions for drug discovery. This project combines chemistry domain knowledge with advanced deep learning architectures to accelerate the identification of potential drug candidates.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Data Processing](#1-data-processing)
  - [2. Model Training](#2-model-training)
  - [3. Running the App](#3-running-the-app)
- [Examples](#examples)
- [Requirements](#requirements)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Introduction

This project leverages Graph Neural Networks to predict interactions between molecules based on their structural features. By representing molecules as graphs—where atoms are nodes and bonds are edges—the GNN can learn complex patterns inherent in molecular structures. This approach aims to aid drug discovery by predicting potential interactions efficiently.

## Features

- **Molecular Graph Representation:** Converts SMILES strings into graph representations for processing.
- **Graph Neural Network Model:** Utilizes GNNs built with PyTorch Geometric for interaction prediction.
- **Interactive Application:** Provides a Streamlit app for real-time prediction and visualization.
- **Visualization:** Displays molecular structures using RDKit.

## Project Structure

- `graph.py`: Processes the dataset and converts SMILES strings to graph data objects.
- `gnn.py`: Defines, trains, and evaluates the GNN model.
- `app.py`: Streamlit application for interactive predictions.
- `sample_data.csv`: Sample dataset containing molecule pairs and interaction labels.
- `processed_data.pt`: Processed data ready for model training.
- `gnn_model.pth`: Saved trained model weights.

## Dataset

Due to the scarcity and protection of molecular interaction data, a synthetic dataset `sample_data.csv` was created for this project. The dataset includes SMILES strings of molecule pairs and binary labels indicating whether they interact (`1` for interaction, `0` for no interaction).

## Installation

### Prerequisites

- Python 3.8 or higher
- `pip` package manager

### Dependencies

Install the required packages using `pip`:

```bash
pip install -r requirements.txt
```

Alternatively, install them individually:

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install pandas scikit-learn
pip install rdkit-pypi
pip install streamlit
```

**Note:** Installing RDKit can be platform-specific. If you encounter issues, refer to the [RDKit Installation Guide](https://www.rdkit.org/docs/Install.html).

### Setting Up a Virtual Environment (Recommended)

Create and activate a virtual environment:

```bash
python3 -m venv gnn_molecule_env
source gnn_molecule_env/bin/activate  # On macOS/Linux
gnn_molecule_env\Scripts\activate     # On Windows
```

## Usage

### 1. Data Processing

Run `graph.py` to process the dataset and generate `processed_data.pt`:

```bash
python graph.py
```

### 2. Model Training

Train the GNN model using `gnn.py`:

```bash
python gnn.py
```

This will train the model and save the weights to `gnn_model.pth`.

### 3. Running the App

Start the Streamlit application using `app.py`:

```bash
streamlit run app.py
```

This will launch the app in your default web browser.

## Examples

Once the app is running, you can input SMILES strings of molecules to predict their interaction.

**Example Input:**

- **Molecule 1 SMILES:** `CCO` (Ethanol)
- **Molecule 2 SMILES:** `CN` (Methylamine)

**Expected Output:**

- **Interaction Probability:** *e.g.,* 0.85
- **Prediction:** *Predicted to interact.*
- **Molecular Structures:** Visualizations of both molecules.

## Requirements

- `torch`
- `torch-geometric`
- `pandas`
- `scikit-learn`
- `rdkit-pypi`
- `streamlit`

## Limitations

- **Data Size:** The model is trained on a small synthetic dataset, which may limit its predictive accuracy.
- **Generalization:** Predictions may not generalize well to molecules significantly different from those in the training data.
- **Data Scarcity:** Due to limited access to large molecular interaction datasets, the model's performance is constrained.

## Future Work

- **Dataset Expansion:** Incorporate larger, real-world datasets to improve model accuracy.
- **Model Optimization:** Experiment with different GNN architectures and hyperparameters.
- **Feature Engineering:** Enhance atom and bond features for better representation.
- **Transfer Learning:** Utilize pre-trained models on molecular data for better performance.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or suggestions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **RDKit** for molecular structure processing and visualization.
- **PyTorch Geometric** for providing tools to build and train GNNs.
- **Streamlit** for creating the interactive web application.
- **Hackathon Challenge:** This project was developed as a weekend challenge to explore innovative solutions in drug discovery.


---
