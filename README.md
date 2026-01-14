# Genomic Mechanistic Interpretability

Mechanistic interpretability analysis of genomic foundation models using attention visualization, activation patching, circuit discovery, and sparse autoencoders.

## Features

- **Multi-Dataset Support**: sQTL, eQTL, ClinVar, GWAS, MAVE, and more
- **4 Analysis Methods**:
  - Attention visualization and pattern analysis
  - Activation patching for causal analysis
  - Circuit discovery and ablation studies
  - Sparse autoencoder for feature learning

## Quick Start

### Installation

```bash
# Clone repository
git clone <repo-url>
cd genomic-mechanistic-interpretability

# Install dependencies
pip install -r requirements.txt

# Setup genomic-FM data loaders (required)
cd ../
git clone https://github.com/ranaabarghout/genomic-FM.git
```

### Basic Usage

**Run complete analysis on sQTL data:**
```bash
python scripts/run_complete_analysis.py --num-samples 200 --train-sae
```

**Run on other datasets:**
```bash
# eQTL (Expression QTL)
python scripts/run_multi_dataset_analysis.py --dataset eqtl --num-samples 200 --train-sae

# ClinVar (Pathogenic variants)
python scripts/run_multi_dataset_analysis.py --dataset clinvar --num-samples 200

# GWAS (Trait-associated variants)
python scripts/run_multi_dataset_analysis.py --dataset gwas --num-samples 200

# MAVE (Experimental variant effects)
python scripts/run_multi_dataset_analysis.py --dataset mave --num-samples 200
```

**Quick test mode (50 samples, ~3 min):**
```bash
python scripts/run_multi_dataset_analysis.py --dataset eqtl --quick-test --train-sae
```

## Project Structure

```
genomic-mechanistic-interpretability/
├── src/
│   ├── data/              # Data loaders for each dataset
│   ├── interpretability/  # Analysis methods (attention, patching, SAE, etc.)
│   └── models/            # Model loading utilities
├── scripts/               # Executable analysis scripts
│   ├── run_complete_analysis.py      # Full sQTL analysis
│   ├── run_multi_dataset_analysis.py # Universal multi-dataset analysis
│   ├── run_attention_analysis.py     # Attention-only analysis
│   ├── run_activation_patching.py    # Patching-only analysis
│   ├── run_circuit_analysis.py       # Circuit discovery only
│   └── run_sparse_autoencoder.py     # SAE training only
├── outputs/               # Analysis results and visualizations
└── notebooks/             # Jupyter notebooks for exploration
```

## Supported Datasets

| Dataset | Description | Command Flag |
|---------|-------------|--------------|
| **sQTL** | Splicing QTL variants | `--dataset sqtl` |
| **eQTL** | Expression QTL variants | `--dataset eqtl` |
| **ClinVar** | Pathogenic/benign variants | `--dataset clinvar` |
| **GWAS** | Trait-associated variants | `--dataset gwas` |
| **MAVE** | Experimental variant effects | `--dataset mave` |

## Analysis Methods

### 1. Attention Visualization
Analyzes attention patterns to understand which sequence positions the model focuses on for variant classification.

### 2. Activation Patching
Causal intervention analysis to identify which model components are critical for predictions.

### 3. Circuit Discovery
Discovers functional circuits of attention heads that work together for specific tasks.

### 4. Sparse Autoencoder
Learns interpretable features from model activations to understand internal representations.

## Output

Each analysis generates:
- **Visualizations**: PNG plots of attention, circuits, features
- **Reports**: Text summaries with key findings
- **Models**: Trained SAE models (if `--train-sae`)
- **Summary**: Overall analysis report

Results are saved to `outputs/<dataset>_analysis/analysis_<timestamp>/`

## Common Options

```bash
--dataset       Dataset to analyze (eqtl, clinvar, gwas, mave)
--num-samples   Number of samples to analyze (default: 200)
--quick-test    Fast test with 50 samples (~3 min)
--train-sae     Train sparse autoencoder (adds ~2 min)
--sae-epochs    Number of SAE training epochs (default: 50)
--output-dir    Custom output directory
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers (Hugging Face)
- DNABERT-2 model (auto-downloaded)
- genomic-FM repository (for data loaders)

See `requirements.txt` for full dependencies.

## Citation

If you use this code, please cite:

```bibtex
@article{your-paper,
  title={Mechanistic Interpretability of Genomic Foundation Models},
  author={Your Name},
  year={2026}
}
```

## License

MIT License - see LICENSE file for details.

