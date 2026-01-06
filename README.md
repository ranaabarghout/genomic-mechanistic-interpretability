# Genomic Mechanistic Interpretability of Foundation Models

**Author:** Rana Barghout  

This repository contains reproducible code and analysis for the technical test:
*Mechanistic Interpretability of Genomic Foundation Models for Genetic Variants*.

## Overview
We analyze how a pretrained genomic foundation model (e.g. DNABERT / Nucleotide Transformer)
interprets genetic variants and predicts functional impact, using mechanistic interpretability tools.

## Repository Structure
```
data/               # scripts to download & preprocess public datasets
models/             # model loading and optional fine-tuning code
interpretability/   # attention, activation patching, SAE, ablations
notebooks/          # exploratory and figure-generation notebooks
figures/            # generated figures for the report
report/             # PDF report source and final PDF
scripts/            # runnable pipelines
```

## Setup
```bash
conda create -n genomic_interp python=3.10
conda activate genomic_interp
pip install -r requirements.txt
```

## Quick Start
```bash
python scripts/run_attention_analysis.py
```

## Data
Public variant datasets are downloaded automatically via scripts in `data/`.

## Reproducibility
All experiments are seed-controlled. See individual scripts for configuration.
