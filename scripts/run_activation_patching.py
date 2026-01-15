#!/usr/bin/env python3
"""
Activation Patching Analysis for sQTL Variants
==============================================
Implements causal intervention experiments on DNABERT-2 to identify which
model components are causally responsible for sQTL significance predictions.

Methods:
1. Layer-wise patching: Test importance of each transformer layer
2. Position-based patching: Test importance of specific sequence positions
3. Attention head patching: Identify critical attention heads
4. Causal tracing: Track information flow through layers

Usage:
    python scripts/run_activation_patching.py --num-samples 50
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from interpretability.activation_patcher import QTLActivationPatcher


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description="Activation Patching Analysis for sQTL Variants",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--num-samples', type=int, default=50,
                       help='Number of samples to analyze (default: 50)')
    parser.add_argument('--output-dir', type=str, default='outputs/activation_patching',
                       help='Output directory (default: outputs/activation_patching)')

    args = parser.parse_args()

    print("="*70)
    print("ACTIVATION PATCHING ANALYSIS FOR sQTL VARIANTS")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Samples: {args.num_samples}")
    print(f"  Output: {args.output_dir}")
    print()

    # Initialize patcher
    patcher = QTLActivationPatcher(
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )

    # Run complete analysis
    results = patcher.run_full_analysis()

    return results


if __name__ == "__main__":
    main()
