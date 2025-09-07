# Final Dissertation Project: Advanced Computational Pathology Methods

This repository contains a comprehensive collection of state-of-the-art computational pathology methods for whole slide image (WSI) analysis, implemented as part of a final dissertation project. The project encompasses multiple deep learning approaches including attention-based multiple instance learning, graph neural networks, and foundation models for pathological image analysis.

## 🔬 Project Overview

This project explores various computational pathology approaches for analyzing whole slide images (WSIs) in cancer diagnosis and prognosis. The repository integrates several cutting-edge methods:

- **CLAM**: Clustering-constrained Attention Multiple Instance Learning for weakly supervised WSI classification
- **Patch-GCN**: Graph Convolutional Networks for context-aware survival prediction 
- **TRIDENT**: Large-scale WSI processing toolkit with foundation model support
- **SUGAR**: Subgraph Neural Network with Reinforcement Learning pooling
- **GigaPath**: Whole slide foundation model for pathology

## 📁 Repository Structure

```
final_dissertation/
├── CLAM/                    # CLAM implementation for WSI classification
├── Patch-GCN/               # Graph-based survival prediction
├── TRIDENT/                 # WSI processing toolkit
├── SUGAR/                   # Subgraph neural networks with RL
├── prov-gigapath/           # GigaPath foundation model
├── TCGA-WSI/                # TCGA dataset processing utilities
├── TCGA-CLAM-graph/         # Processed TCGA data for graph methods
├── TCGA-patches/            # Extracted patches from TCGA
├── trident_processed/       # Processed WSI data using TRIDENT
└── test/                    # Test data and validation files
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Minimum 16GB RAM
- Large storage capacity for WSI data

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd final_dissertation
   ```

2. **Set up individual environments**
   
   Each method has its own environment requirements. Choose the method you want to use:

   **For CLAM:**
   ```bash
   cd CLAM
   conda env create -f env.yml
   conda activate clam
   ```

   **For TRIDENT:**
   ```bash
   cd TRIDENT
   conda create -n trident python=3.10
   conda activate trident
   pip install -e .
   ```

   **For SUGAR:**
   ```bash
   cd SUGAR
   pip install -r requirements.txt
   ```

   **For GigaPath:**
   ```bash
   cd prov-gigapath
   conda env create -f environment.yaml
   conda activate gigapath
   ```

## 🔧 Usage

### 1. CLAM - Weakly Supervised WSI Classification

CLAM enables slide-level classification without patch-level annotations.

```bash
cd CLAM

# Segmentation and patching
python create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 256 --seg --patch --stitch

# Feature extraction
python extract_features_fp.py --data_h5_dir RESULTS_DIRECTORY --data_slide_dir DATA_DIRECTORY --csv_path dataset_csv/tumor_vs_normal_dummy_clean.csv --feat_dir FEATURES_DIRECTORY

# Training
python main.py --drop_out --early_stopping --lr 2e-4 --k 10 --label_frac 1.0 --exp_code tumor_vs_normal_CLAM_50 --weighted_sample --bag_loss ce --inst_loss svm --task tumor_vs_normal --model_type clam_sb --log_data --data_root_dir FEATURES_DIRECTORY
```

### 2. Patch-GCN - Graph-based Survival Prediction

Formulates WSIs as graphs for context-aware analysis.

```bash
cd Patch-GCN

# WSI-Graph Construction
python "WSI-Graph Construction.py"

# Training
python main.py --task tcga_kidney_cv --split_dir tcga_kidney_100 --gc 32
```

### 3. TRIDENT - WSI Processing Toolkit

High-performance WSI processing with foundation model support.

```bash
cd TRIDENT

# Process single slide
python run_single_slide.py --slide_path ./wsis/slide.svs --job_dir ./processed --patch_encoder uni_v1 --mag 20 --patch_size 256

# Batch processing
python run_batch_of_slides.py --task all --wsi_dir ./wsis --job_dir ./processed --patch_encoder uni_v1 --mag 20 --patch_size 256
```

### 4. SUGAR - Subgraph Neural Networks

Graph neural networks with reinforcement learning pooling.

```bash
cd SUGAR
python train.py
```

## 📊 Datasets

The project works with several major pathology datasets:

- **TCGA (The Cancer Genome Atlas)**: Multi-cancer WSI dataset
  - BLCA (Bladder Cancer)
  - BRCA (Breast Cancer) 
  - LUAD (Lung Adenocarcinoma)
  - UCEC (Uterine Corpus Endometrial Carcinoma)

- **Graph Datasets**: MUTAG, DD, NCI1, NCI109, PTC_MR, ENZYMES, PROTEINS

### Data Organization

```
TCGA-WSI/
├── BLCA/           # Bladder cancer WSIs
├── BRCA/           # Breast cancer WSIs  
├── LUAD/           # Lung cancer WSIs
└── UCEC/           # Uterine cancer WSIs

TCGA-patches/       # Extracted tissue patches
TCGA-CLAM-graph/    # Graph representations
```

## 🏗️ Key Features

### CLAM Features
- Weakly supervised learning using only slide-level labels
- Attention-based multiple instance learning
- Clustering-constrained feature learning
- Heatmap visualization for interpretability

### Patch-GCN Features  
- Graph representation of WSI spatial relationships
- k-NN connectivity based on patch coordinates
- Context-aware survival prediction
- Message passing for spatial feature learning

### TRIDENT Features
- Support for 25+ foundation models (UNI, CONCH, TITAN, etc.)
- Efficient tissue segmentation and patch extraction
- Scalable processing pipeline
- Multiple encoder options

### SUGAR Features
- Reinforcement learning-based graph pooling
- Self-supervised mutual information mechanism
- Subgraph neural network architecture

## 📈 Results and Evaluation

The project includes comprehensive evaluation scripts and benchmarks:

- Classification accuracy metrics
- Survival prediction C-index
- Attention visualization and interpretation
- Cross-validation results
- Statistical significance testing

## 📚 Documentation

Detailed documentation for each method can be found in their respective directories:

- [`CLAM/docs/`](CLAM/docs/) - CLAM documentation and tutorials
- [`TRIDENT/docs/`](TRIDENT/docs/) - TRIDENT user guide
- [`Patch-GCN/docs/`](Patch-GCN/docs/) - Patch-GCN implementation details

## 🤝 Contributing

This is a research project developed for academic purposes. For questions or collaboration opportunities, please feel free to reach out.

## 📄 License

Each component retains its original license:
- CLAM: GPLv3 License
- TRIDENT: MIT License  
- Patch-GCN: Academic use
- SUGAR: Academic use

## 🔗 References

If you use this work, please cite the relevant papers:

### CLAM
```bibtex
@article{lu2021data,
  title={Data-efficient and weakly supervised computational pathology on whole-slide images},
  author={Lu, Ming Y and Williamson, Drew FK and Chen, Tiffany Y and Chen, Richard J and Barbieri, Matteo and Mahmood, Faisal},
  journal={Nature biomedical engineering},
  volume={5},
  number={6},
  pages={555--570},
  year={2021},
  publisher={Nature Publishing Group}
}
```

### Patch-GCN
```bibtex
@incollection{chen2021whole,
  title={Whole Slide Images are 2D Point Clouds: Context-Aware Survival Prediction using Patch-based Graph Convolutional Networks},
  author={Chen, Richard J and Lu, Ming Y and Shaban, Muhammad and Chen, Chengkuan and Chen, Tiffany Y and Williamson, Drew FK and Mahmood, Faisal},
  booktitle={Medical Image Computing and Computer Assisted Intervention--MICCAI 2021},
  pages={339--349},
  year={2021},
  publisher={Springer}
}
```

### TRIDENT
```bibtex
@article{chen2025trident,
  title={Trident: A foundation model for digital pathology},
  author={Chen, Richard J and others},
  journal={arXiv preprint arXiv:2502.06750},
  year={2025}
}
```

---

**Developed as part of UCL Final Dissertation Project**  
*Advancing Computational Pathology through Deep Learning and Graph Neural Networks*
