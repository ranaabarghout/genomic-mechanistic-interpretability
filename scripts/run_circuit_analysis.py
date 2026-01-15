#!/usr/bin/env python3
"""
Circuit Analysis and Ablation for sQTL Variants
===============================================
Discover and ablate functional circuits in DNABERT-2 that work together
to classify sQTL significance.

A "circuit" is a group of neurons, attention heads, or layers that work
together to implement a specific computational function. This script:

1. Circuit Discovery: Find groups of attention heads that consistently
   activate together for significant vs not_significant sQTLs

2. Systematic Ablation: Disable discovered circuits and measure impact
   on classification performance

3. Layer-wise Ablation: Test contribution of each layer by zeroing outputs

4. Head-wise Ablation: Test contribution of individual attention heads

Usage:
    python scripts/run_circuit_analysis.py --num-samples 100
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from interpretability.circuit_analyzer import CircuitAnalyzer


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description="Circuit Analysis and Ablation for sQTL Variants",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of samples to analyze (default: 100)')
    parser.add_argument('--output-dir', type=str, default='outputs/circuit_analysis',
                       help='Output directory (default: outputs/circuit_analysis)')

    args = parser.parse_args()

    print("="*70)
    print("CIRCUIT ANALYSIS AND ABLATION FOR sQTL VARIANTS")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Samples: {args.num_samples}")
    print(f"  Output: {args.output_dir}")
    print()

    # Initialize analyzer
    analyzer = CircuitAnalyzer(
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )

    # Run complete analysis
    results = analyzer.run_full_analysis()

    return results


if __name__ == "__main__":
    main()
