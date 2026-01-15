# Genomic Mechanistic Interpretability

Mechanistic interpretability analysis of genomic foundation models using attention visualization, activation patching, circuit discovery, and sparse autoencoders. This project analyzes **DNA-BERT-6** on **sQTL** (splicing QTL) and **eQTL** (expression QTL) variants to understand how the model processes functional genomic variants.

## Features

- **Model**: DNA-BERT-6 (6-layer BERT model trained on genomic sequences)
- **Primary Datasets**: sQTL and eQTL variants from GTEx
- **3 Analysis Methods**:
  - Attention visualization and pattern analysis
  - Activation patching for causal analysis
  - Sparse autoencoder for feature learning

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/ranaabarghout/genomic-mechanistic-interpretability.git
cd genomic-mechanistic-interpretability

# Install dependencies
pip install -r requirements.txt

# Setup genomic-FM data loaders (required)
cd ../
git clone https://github.com/ranaabarghout/genomic-FM.git
```

### Basic Usage

**Run complete analysis on sQTL data (1000 samples):**
```bash
python scripts/run_multi_dataset_analysis.py --dataset sqtl --num-samples 1000 --mechanistic-attention --train-sae --sae-epochs 50
```

**Run complete analysis on eQTL data (1000 samples):**
```bash
python scripts/run_multi_dataset_analysis.py --dataset eqtl --num-samples 1000 --mechanistic-attention --train-sae --sae-epochs 50
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
│   ├── interpretability/  # Analysis modules (attention, patching, SAE, circuit)
│   └── models/            # Model loading utilities
├── scripts/               # Executable analysis scripts
│   ├── run_multi_dataset_analysis.py    # Main unified analysis pipeline
│   ├── run_hidden_state_analysis.py     # Hidden state analysis only
│   ├── run_mechanistic_attention_analysis.py  # Attention analysis only
│   ├── run_activation_patching.py       # Activation patching only
│   ├── run_circuit_analysis.py          # Circuit discovery only
│   ├── run_sparse_autoencoder.py        # SAE training only
│   └── archive/                          # Legacy scripts
├── outputs/               # Analysis results and visualizations
│   ├── eqtl_analysis/    # eQTL analysis outputs
│   └── sqtl_analysis/    # sQTL analysis outputs
├── report/                # LaTeX report template and figures
├── docs/                  # Documentation
└── root/data/             # Data storage directory
```

## Supported Datasets

| Dataset | Description | Status |
|---------|-------------|--------|
| **sQTL** | Splicing QTL variants from GTEx | ✅ Available |
| **eQTL** | Expression QTL variants from GTEx | ✅ Available |
| **ClinVar** | Pathogenic/benign variants | 🔜 Future support |
| **GWAS** | Trait-associated variants | 🔜 Future support |
| **MAVE** | Experimental variant effects | 🔜 Future support |

Currently, the analysis pipeline is optimized for **sQTL** and **eQTL** variants. Support for additional datasets (ClinVar, GWAS, MAVE) will be added in future releases.

## Analysis Methods

### 1. Attention Visualization
Analyzes attention patterns to understand which sequence positions the model focuses on for variant classification.

### 2. Activation Patching
Causal intervention analysis to identify which model components are critical for predictions.

### 3. Sparse Autoencoder
Learns interpretable features from model activations to understand internal representations.

### 4. Circuit Discovery (Work in Progress)
Will discover functional circuits of attention heads that work together for specific tasks. This analysis method will be available in future releases.

## Output

Each analysis generates:
- **Visualizations**: PNG plots of attention, circuits, features
- **Reports**: Text summaries with key findings
- **Models**: Trained SAE models (if `--train-sae`)
- **Summary**: Overall analysis report

Results are saved to `outputs/<dataset>_analysis/analysis_<timestamp>/`

## Common Options

```bash
--dataset            Dataset to analyze (sqtl, eqtl, clinvar, gwas, mave)
--num-samples        Number of samples to analyze (default: 200)
--mechanistic-attention  Use enhanced mechanistic attention analysis
--train-sae          Train sparse autoencoder (adds ~2 min)
--sae-epochs         Number of SAE training epochs (default: 50)
--run-circuit        Run circuit analysis (NOT CURRENTLY RECOMMENDED, optional)
--quick-test         Fast test with 50 samples (~3 min)
--output-dir         Custom output directory
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers (Hugging Face)
- DNA-BERT-6 model (zhihan1996/DNA_bert_6, auto-downloaded)
- genomic-FM repository (for data loaders)

See `requirements.txt` for full dependencies. Please also see the genomic-FM dependency instructions for additional packages that might be needed!

## Citation

If you use this code, please cite:

```bibtex
@software{barghout2026genomic,
  title={Genomic Mechanistic Interpretability},
  author={Barghout, Rana A.},
  year={2026},
  url={https://github.com/ranaabarghout/genomic-mechanistic-interpretability}
}
```

## License

MIT License - see LICENSE file for details.

