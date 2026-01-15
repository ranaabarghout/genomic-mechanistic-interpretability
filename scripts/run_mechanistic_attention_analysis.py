#!/usr/bin/env python3
"""
Mechanistic Attention Analysis for Genomic Variants
===================================================
Enhanced attention analysis with mechanistic interpretability approach:
1. Token-specific attention (CLS, variant-centered queries)
2. Correct variant position mapping after tokenization
3. Per-head and per-layer analysis with entropy/variance metrics
4. Class-conditioned statistical comparisons
5. Integration with functional importance (ablation/patching)

Usage:
    python scripts/run_mechanistic_attention_analysis.py --dataset eqtl --num-samples 100
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from interpretability.mechanistic_attention_analyzer import MechanisticAttentionAnalyzer


def main():
    parser = argparse.ArgumentParser(description="Mechanistic attention analysis")
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (eqtl, sqtl, clinvar, etc.)')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of samples to analyze')
    parser.add_argument('--model', type=str, default="zhihan1996/DNABERT-2-117M",
                        help='Model name')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')

    args = parser.parse_args()

    analyzer = MechanisticAttentionAnalyzer(
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        model_name=args.model,
        output_dir=args.output_dir
    )

    results = analyzer.run_full_mechanistic_analysis()


if __name__ == "__main__":
    main()
