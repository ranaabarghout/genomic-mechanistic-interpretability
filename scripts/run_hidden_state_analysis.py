#!/usr/bin/env python3
"""
Hidden State Analysis for Genomic Variants
==========================================
Mechanistic interpretability using hidden state representations:
1. Layer-wise hidden state evolution
2. Variant-centered representation analysis
3. Class-conditioned statistical comparisons
4. Dimensionality reduction (PCA, t-SNE) for visualization
5. Integration with activation patching

Note: This replaces attention-based analysis for models like DNABERT-2
that don't expose attention weights.

Usage:
    python scripts/run_hidden_state_analysis.py --dataset eqtl --num-samples 100
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from interpretability.hidden_state_analyzer import HiddenStateAnalyzer


def main():
    parser = argparse.ArgumentParser(description="Hidden state analysis")
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (eqtl, sqtl, clinvar, etc.)')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of samples to analyze')
    parser.add_argument('--model', type=str, default="zhihan1996/DNABERT-2-117M",
                        help='Model name')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')

    args = parser.parse_args()

    analyzer = HiddenStateAnalyzer(
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        model_name=args.model,
        output_dir=args.output_dir
    )

    results = analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
