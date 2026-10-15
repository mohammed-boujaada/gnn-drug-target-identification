
<div align="center">

# GNN-Based Drug Target Identification

### Sparse Graph Attention and Structured Neuron Pruning for Disease Gene Classification

*Master's Research Project in Artificial Intelligence and Emerging Technologies (MIATE)*
*Faculty of Applied Sciences, Nador - Mohammed First University*

<br/>

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-3C2179?style=for-the-badge&logo=pyg&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NetworkX](https://img.shields.io/badge/NetworkX-Graphs-4B8BBE?style=for-the-badge)

![STRING](https://img.shields.io/badge/Data-STRING%20v12.0-0F766E?style=flat-square)
![DisGeNET](https://img.shields.io/badge/Data-DisGeNET-0369A1?style=flat-square)
![Disease](https://img.shields.io/badge/Target%20Disease-Diabetes%20Mellitus-B91C1C?style=flat-square)
![Task](https://img.shields.io/badge/Task-Node%20Classification-6D28D9?style=flat-square)
![License](https://img.shields.io/badge/License-Educational-orange?style=flat-square)
![Status](https://img.shields.io/badge/Status-Completed-success?style=flat-square)

</div>

---

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Key Results](#2-key-results)
- [3. Methodology](#3-methodology)
- [4. Data Splitting Strategy](#4-data-splitting-strategy)
- [5. Results Visualization](#5-results-visualization)
- [6. Repository Structure](#6-repository-structure)
- [7. Installation and Execution](#7-installation-and-execution)
- [8. Dataset Sources](#8-dataset-sources)
- [9. Limitations](#9-limitations)
- [10. Citation](#10-citation)
- [11. Acknowledgments](#11-acknowledgments)
- [12. Authors](#12-authors)

---

## 1. Project Overview

This repository contains the complete implementation of our research on **Graph Neural Networks (GNNs) for disease-gene classification** on a human protein-protein interaction (PPI) network, with **Diabetes Mellitus** as the target disease.

The study addresses a typical problem in computational drug discovery: identifying candidate disease genes in a large biological network where known positive examples are extremely rare (0.47% of nodes). It compares a standard Graph Attention Network (GAT) with a sparse variant (SGAT) that learns which edges to keep, then compresses the model through structured neuron pruning.

The entire pipeline (data preprocessing, GAT baseline, SGAT sparse training, structured neuron pruning, and disease-gene ranking) is implemented in a **single Jupyter notebook**: [`gnn_drug_target_identification.ipynb`](gnn_drug_target_identification.ipynb).

```mermaid
flowchart LR
    A[STRING PPI network<br/>score >= 700] --> C[Graph construction<br/>15,882 nodes]
    B[DisGeNET<br/>Diabetes Mellitus genes] --> D[Node labels<br/>74 positives]
    C --> E[Node features<br/>degree, clustering, PageRank]
    D --> E
    E --> F[Baseline GAT<br/>5 seeds]
    E --> G[SGAT<br/>learnable edge gates + L0]
    G --> H[Structured neuron pruning<br/>128 to 28 neurons]
    F --> I[Evaluation<br/>ROC-AUC, PR-AUC]
    H --> I
    H --> J[Disease-gene ranking]
```

---

## 2. Key Results

| Model | Neurons | Test ROC-AUC | Test PR-AUC | Edge Sparsity |
|---|---|---|---|---|
| Baseline GAT (5 seeds) | 128 | 0.6860 ± 0.0601 | 0.0093 ± 0.0020 | 0% |
| SGAT | 128 | 0.6817 | 0.0167 | 71.48% |
| SGAT compact backbone | 28 | 0.7140 | 0.0142 | 71.48% |

<div align="center">

![Edge Sparsity](https://img.shields.io/badge/Edge%20Sparsity-71.48%25-blue?style=for-the-badge)
![Neuron Reduction](https://img.shields.io/badge/Neuron%20Reduction-78.12%25-green?style=for-the-badge)
![Compact ROC-AUC](https://img.shields.io/badge/Compact%20ROC--AUC-0.7140-purple?style=for-the-badge)

</div>

The compact model reduces the hidden representation by **78.12%** (128 to 28 neurons) while retaining the learned **71.48%** edge sparsity.

---

## 3. Methodology

| Step | Stage | Description |
|---|---|---|
| 1 | **Graph construction** | STRING high-confidence interactions (combined score >= 700): 15,882 nodes and 236,712 undirected edges. |
| 2 | **Disease labels** | DisGeNET Diabetes Mellitus genes: 74 mapped positive nodes versus 15,808 negative nodes (0.47% prevalence). |
| 3 | **Node features** | Degree, clustering coefficient, and PageRank. |
| 4 | **Baseline GAT** | Two-layer graph attention network, evaluated across 5 random seeds. |
| 5 | **SGAT adaptation** | Learnable edge gates with L0 sparsity regularization (lambda = 1e-6). |
| 6 | **Structured neuron pruning** | Importance ranking of the 128 hidden neurons; the smallest representation satisfying a 95% validation ROC-AUC retention criterion contains 28 neurons. |
| 7 | **Disease-gene ranking** | Computational prioritization of candidate genes (EGFR, CTNNB1, TP53, among others); known genes FN1, RAC1, and INS are recovered in the top 30. |

### Model Architecture

<div align="center">

<img src="model_architecture.jfif" alt="Model architecture: GAT versus SGAT" width="85%"/>

</div>

---

## 4. Data Splitting Strategy

<div align="center">

<img src="data_split.jfif" alt="Stratified data split" width="85%"/>

</div>

A stratified 60/20/20 split (seed 42) preserves the class ratio across subsets:

| Subset | Nodes | Positive examples |
|---|---|---|
| Train | 9,529 | 44 |
| Validation | 3,176 | 15 |
| Test | 3,177 | 15 |

---

## 5. Results Visualization

### Neuron Destruction Curve (Validation ROC-AUC)

<div align="center">

<img src="sgat_neuron_destruction_roc_auc.png" alt="SGAT neuron destruction curve (ROC-AUC)" width="80%"/>

</div>

The 28-neuron configuration is the smallest tested representation that satisfies the 95% validation ROC-AUC retention threshold (0.7315).

### Neuron Pruning Curve (PR-AUC)

<div align="center">

<img src="sgat_neuron_pruning_pr_auc.png" alt="SGAT neuron pruning curve (PR-AUC)" width="80%"/>

</div>

---

## 6. Repository Structure

```text
gnn-drug-target-identification/
│
│
│── data_split.jfif                          # Data splitting visualization
├── model_architecture.jfif                  # Model architecture diagram
│
├
├── sgat_neuron_destruction_roc_auc.png  # Neuron destruction curve (ROC-AUC)
│── sgat_neuron_pruning_pr_auc.png       # Neuron pruning curve (PR-AUC)│
│── .gitignore
├── gnn_drug_target_identification.ipynb     # Complete pipeline (single notebook)
├── README.md
└── requirements.txt
```

**Note:** The `data/` folder is not included in the repository due to file size limitations. Download the datasets separately using the links provided in the [Dataset Sources](#8-dataset-sources) section.

---

## 7. Installation and Execution

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook
- Dependencies listed in `requirements.txt`

### Setup

```bash
git clone https://github.com/mohammed-boujaada/gnn-drug-target-identification.git
cd gnn-drug-target-identification

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Download Datasets

Create a `data/` folder and download the required datasets (see [Dataset Sources](#8-dataset-sources) below).

### Run the Pipeline

```bash
jupyter notebook
```

Open `gnn_drug_target_identification.ipynb` and execute the cells in order. The notebook performs the following steps sequentially:

1. Graph construction and label mapping
2. Node feature computation
3. GAT baseline (5 seeds)
4. SGAT training with learnable edge gates
5. Sparsity analysis
6. Structured neuron pruning
7. Final disease-gene ranking

---

## 8. Dataset Sources

### Download Link

📥 **[Download all datasets here](https://www.kaggle.com/datasets)** *(or replace with your actual Kaggle/Google Drive link)*

Alternatively, download each dataset separately:

| Dataset | Version | Used for | Link |
|---|---|---|---|
| **STRING** | v12.0 (Homo sapiens, taxon 9606) | Protein information and interactions, filtered at combined score >= 700 | [string-db.org](https://string-db.org/cgi/download.pl?species_text=Homo+sapiens) |
| **DisGeNET** | Curated gene-disease associations | Diabetes Mellitus gene associations (91 genes, 74 mapped to the PPI graph) | [disgenet.org](https://www.disgenet.org/downloads) |

**Instructions:**
1. Create a folder named `data/` in the root directory
2. Download the STRING files for Homo sapiens (taxon 9606):
   - `9606.protein.info.v12.0.txt`
   - `9606.protein.links.v12.0.txt`
3. Download the DisGeNET curated gene-disease associations:
   - `curated_gene_disease_associations.tsv`
4. Place all three files in the `data/` folder

---

## 9. Limitations

- Strong class imbalance (0.47% positive nodes): evaluation relies on ROC-AUC and PR-AUC rather than accuracy.
- Topology-based features only; no sequence, expression, or pathway information is used.
- 17 DisGeNET genes could not be mapped to the PPI graph.
- Negative labels do not prove biological irrelevance to the disease.
- The gene ranking provides computational hypotheses, not validated drug targets.

---

## 10. Citation

```bibtex
@mastersthesis{zahir2026gnn,
  title  = {GNN-Based Drug Target Identification: Sparse Graph Attention
            and Structured Neuron Pruning for Disease Gene Classification},
  author = {Zahir, Mohamed and Boujaada, Mohammed},
  school = {FSA Nador, Universit\'e Mohammed Premier},
  year   = {2026}
}
```

---

## 11. Acknowledgments

The authors thank Prof. Khalid El Makkaoui and Prof. Ibrahim Ouahbi for their coordination and academic support, and Prof. Jamal Boussouf for his mentorship, guidance, and valuable feedback throughout this work.

---

## 12. Authors

<div align="center">

<table>
<tr>
<td align="center" width="260">
<a href="https://github.com/mohammed-boujaada">
<img src="https://github.com/mohammed-boujaada.png" width="120px" alt="Mohammed Boujaada"/>
<br/>
<sub><b>Mohammed Boujaada</b></sub>
</a>
<br/>
MIATE Master's, FSA Nador
<br/><br/>
<a href="https://github.com/mohammed-boujaada">
<img src="https://img.shields.io/badge/GitHub-@mohammed--boujaada-181717?style=flat-square&logo=github"/>
</a>
<br/>
<a href="https://www.linkedin.com/in/mohammed-boujaada-229a3333a/">
<img src="https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin&logoColor=white"/>
</a>
</td>
<td align="center" width="260">
<a href="https://github.com/moza369">
<img src="https://github.com/moza369.png" width="120px" alt="Mohamed Zahir"/>
<br/>
<sub><b>Mohamed Zahir</b></sub>
</a>
<br/>
MIATE Master's, FSA Nador
<br/><br/>
<a href="https://github.com/moza369">
<img src="https://img.shields.io/badge/GitHub-@moza369-181717?style=flat-square&logo=github"/>
</a>
</td>
</tr>
</table>

<br/>

![Last Updated](https://img.shields.io/badge/Last%20Updated-October%202026-lightgrey?style=flat-square)

</div>
