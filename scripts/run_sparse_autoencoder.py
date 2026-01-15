#!/usr/bin/env python3
"""
Sparse Autoencoder Analysis for sQTL Variants
=============================================
Train a sparse autoencoder on DNABERT-2 activations to discover interpretable
features that distinguish significant from not_significant sQTLs.

Key Methods:
1. Train SAE on DNABERT-2 hidden states
2. Identify interpretable latent features
3. Analyze feature activation patterns by sQTL class
4. Visualize most discriminative features

Usage:
    python scripts/run_sparse_autoencoder.py --num-samples 200 --train-sae
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from interpretability.sae_analyzer import QTLSAEAnalyzer


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description="Sparse Autoencoder Analysis for sQTL Variants",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--num-samples', type=int, default=200,
                       help='Number of samples to analyze (default: 200)')
    parser.add_argument('--sae-hidden-dim', type=int, default=2048,
                       help='SAE hidden dimension (default: 2048)')
    parser.add_argument('--sparsity-coef', type=float, default=0.1,
                       help='Sparsity penalty coefficient (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs (default: 50)')
    parser.add_argument('--train-sae', action='store_true',
                       help='Train new SAE (otherwise load existing)')
    parser.add_argument('--output-dir', type=str, default='outputs/sparse_autoencoder',
                       help='Output directory (default: outputs/sparse_autoencoder)')

    args = parser.parse_args()

    print("="*70)
    print("SPARSE AUTOENCODER ANALYSIS FOR sQTL VARIANTS")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Samples: {args.num_samples}")
    print(f"  SAE hidden dim: {args.sae_hidden_dim}")
    print(f"  Sparsity coef: {args.sparsity_coef}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Train SAE: {args.train_sae}")
    print(f"  Output: {args.output_dir}")
    print()

    # Initialize analyzer
    analyzer = QTLSAEAnalyzer(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        sae_hidden_dim=args.sae_hidden_dim,
        sparsity_coef=args.sparsity_coef
    )

    # Run complete analysis
    results = analyzer.run_full_analysis(
        train_sae=args.train_sae,
        epochs=args.epochs
    )

    return results


if __name__ == "__main__":
    main()
