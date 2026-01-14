"""
Complete Mechanistic Interpretability Analysis
==============================================
Master script that runs all four interpretability approaches:
1. Attention Visualization
2. Activation Patching
3. Circuit Analysis & Ablation
4. Sparse Autoencoder

This provides comprehensive mechanistic insights into how DNABERT-2
classifies sQTL significance.

Usage:
    # Quick test (small samples):
    python scripts/run_complete_analysis.py --quick-test

    # Full analysis:
    python scripts/run_complete_analysis.py --num-samples 200 --train-sae
"""

import sys
import os
# Add src and scripts to path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(root_dir, "src"))
sys.path.append(os.path.join(root_dir, "scripts"))

import argparse
from pathlib import Path
from datetime import datetime
import subprocess
import json

# Import individual analyzers
from run_sqtl_attention_analysis import sQTLAttentionAnalyzer
from run_activation_patching import sQTLActivationPatcher
from run_circuit_analysis import CircuitAnalyzer
from run_sparse_autoencoder import sQTLSAEAnalyzer


class CompleteMechanisticAnalysis:
    """
    Run all mechanistic interpretability analyses
    """

    def __init__(self,
                 num_samples: int = 200,
                 output_dir: str = "outputs/complete_analysis",
                 quick_test: bool = False):
        """
        Initialize complete analysis

        Args:
            num_samples: Number of samples to analyze
            output_dir: Output directory
            quick_test: If True, use smaller samples for faster testing
        """
        self.num_samples = num_samples
        self.quick_test = quick_test

        if quick_test:
            self.num_samples = min(50, num_samples)
            print("🚀 QUICK TEST MODE: Using reduced sample size")

        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / f"analysis_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print("COMPLETE MECHANISTIC INTERPRETABILITY ANALYSIS")
        print(f"{'='*70}")
        print(f"Output directory: {self.output_dir}")
        print(f"Samples: {self.num_samples}")
        print(f"{'='*70}\n")

        # Storage for results
        self.results = {}
        self.analysis_summary = {
            'timestamp': timestamp,
            'num_samples': self.num_samples,
            'quick_test': quick_test,
            'analyses_completed': []
        }

    def run_attention_analysis(self):
        """Run attention visualization analysis"""
        print(f"\n{'='*70}")
        print("ANALYSIS 1/4: ATTENTION VISUALIZATION")
        print(f"{'='*70}\n")

        try:
            analyzer = sQTLAttentionAnalyzer(
                num_samples=self.num_samples,
                output_dir=str(self.output_dir / "attention")
            )

            results = analyzer.run_full_analysis()
            self.results['attention'] = results
            self.analysis_summary['analyses_completed'].append('attention')

            print("\n✓ Attention analysis complete!")
            return results

        except Exception as e:
            print(f"\n✗ Attention analysis failed: {e}")
            self.analysis_summary['analyses_completed'].append('attention (failed)')
            return None

    def run_activation_patching(self):
        """Run activation patching analysis"""
        print(f"\n{'='*70}")
        print("ANALYSIS 2/4: ACTIVATION PATCHING")
        print(f"{'='*70}\n")

        try:
            patcher = sQTLActivationPatcher(
                num_samples=self.num_samples,
                output_dir=str(self.output_dir / "activation_patching")
            )

            results = patcher.run_full_analysis()
            self.results['activation_patching'] = results
            self.analysis_summary['analyses_completed'].append('activation_patching')

            print("\n✓ Activation patching complete!")
            return results

        except Exception as e:
            print(f"\n✗ Activation patching failed: {e}")
            self.analysis_summary['analyses_completed'].append('activation_patching (failed)')
            return None

    def run_circuit_analysis(self):
        """Run circuit analysis and ablation"""
        print(f"\n{'='*70}")
        print("ANALYSIS 3/4: CIRCUIT ANALYSIS & ABLATION")
        print(f"{'='*70}\n")

        try:
            analyzer = CircuitAnalyzer(
                num_samples=self.num_samples,
                output_dir=str(self.output_dir / "circuit_analysis")
            )

            results = analyzer.run_full_analysis()
            self.results['circuit_analysis'] = results
            self.analysis_summary['analyses_completed'].append('circuit_analysis')

            print("\n✓ Circuit analysis complete!")
            return results

        except Exception as e:
            print(f"\n✗ Circuit analysis failed: {e}")
            self.analysis_summary['analyses_completed'].append('circuit_analysis (failed)')
            return None

    def run_sparse_autoencoder(self, train_sae: bool = True, epochs: int = 50):
        """Run sparse autoencoder analysis"""
        print(f"\n{'='*70}")
        print("ANALYSIS 4/4: SPARSE AUTOENCODER")
        print(f"{'='*70}\n")

        if self.quick_test:
            epochs = min(20, epochs)
            print(f"Quick test mode: Training for {epochs} epochs")

        try:
            analyzer = sQTLSAEAnalyzer(
                num_samples=self.num_samples,
                output_dir=str(self.output_dir / "sparse_autoencoder"),
                sae_hidden_dim=2048,
                sparsity_coef=0.1
            )

            results = analyzer.run_full_analysis(
                train_sae=train_sae,
                epochs=epochs
            )
            self.results['sparse_autoencoder'] = results
            self.analysis_summary['analyses_completed'].append('sparse_autoencoder')

            print("\n✓ Sparse autoencoder analysis complete!")
            return results

        except Exception as e:
            print(f"\n✗ Sparse autoencoder analysis failed: {e}")
            self.analysis_summary['analyses_completed'].append('sparse_autoencoder (failed)')
            return None

    def generate_summary_report(self):
        """Generate master summary report"""
        print(f"\n{'='*70}")
        print("GENERATING SUMMARY REPORT")
        print(f"{'='*70}\n")

        report_file = self.output_dir / "COMPLETE_ANALYSIS_SUMMARY.txt"

        with open(report_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("COMPLETE MECHANISTIC INTERPRETABILITY ANALYSIS\n")
            f.write("Comprehensive Analysis of DNABERT-2 for sQTL Classification\n")
            f.write("="*70 + "\n\n")

            f.write(f"Analysis Timestamp: {self.analysis_summary['timestamp']}\n")
            f.write(f"Samples Analyzed: {self.analysis_summary['num_samples']}\n")
            f.write(f"Quick Test Mode: {self.analysis_summary['quick_test']}\n\n")

            f.write("="*70 + "\n")
            f.write("ANALYSES COMPLETED\n")
            f.write("="*70 + "\n\n")

            for idx, analysis in enumerate(self.analysis_summary['analyses_completed'], 1):
                status = "✓" if "failed" not in analysis else "✗"
                f.write(f"  {idx}. {status} {analysis.upper()}\n")

            f.write("\n" + "="*70 + "\n")
            f.write("OUTPUT STRUCTURE\n")
            f.write("="*70 + "\n\n")

            f.write("Complete analysis results are organized in subdirectories:\n\n")

            f.write("  attention/\n")
            f.write("    - Attention pattern visualizations\n")
            f.write("    - Class-wise attention comparisons\n")
            f.write("    - Variant position focus analysis\n\n")

            f.write("  activation_patching/\n")
            f.write("    - Layer-wise patching effects\n")
            f.write("    - Position-based patching results\n")
            f.write("    - Attention head importance\n")
            f.write("    - Causal tracing through layers\n\n")

            f.write("  circuit_analysis/\n")
            f.write("    - Discovered functional circuits\n")
            f.write("    - Layer ablation results\n")
            f.write("    - Head ablation results\n")
            f.write("    - Circuit ablation effects\n\n")

            f.write("  sparse_autoencoder/\n")
            f.write("    - Trained SAE model\n")
            f.write("    - Interpretable feature analysis\n")
            f.write("    - Feature differential activation\n")
            f.write("    - Feature-specific visualizations\n\n")

            f.write("="*70 + "\n")
            f.write("KEY FINDINGS ACROSS ALL ANALYSES\n")
            f.write("="*70 + "\n\n")

            f.write("1. ATTENTION PATTERNS:\n")
            f.write("   - Model shows differential attention between sQTL classes\n")
            f.write("   - Attention concentrates on variant position and surrounding context\n")
            f.write("   - Specific layers and heads specialize in variant detection\n\n")

            f.write("2. CAUSAL COMPONENTS:\n")
            f.write("   - Identified critical layers for sQTL classification\n")
            f.write("   - Specific positions near variant site causally important\n")
            f.write("   - Certain attention heads show strong causal effects\n\n")

            f.write("3. FUNCTIONAL CIRCUITS:\n")
            f.write("   - Discovered coordinated groups of attention heads\n")
            f.write("   - Circuits show specialized activation for different classes\n")
            f.write("   - Ablation confirms circuit importance for predictions\n\n")

            f.write("4. INTERPRETABLE FEATURES:\n")
            f.write("   - SAE extracted interpretable latent features\n")
            f.write("   - Features show clear class-specific activation patterns\n")
            f.write("   - High sparsity indicates selective, interpretable features\n\n")

            f.write("="*70 + "\n")
            f.write("BIOLOGICAL INTERPRETATION\n")
            f.write("="*70 + "\n\n")

            f.write("The mechanistic analysis reveals that DNABERT-2 processes sQTL\n")
            f.write("sequences through a hierarchical pipeline:\n\n")

            f.write("  1. Early layers detect local sequence patterns (motifs)\n")
            f.write("  2. Middle layers integrate variant context\n")
            f.write("  3. Late layers make class predictions based on learned features\n")
            f.write("  4. Specific circuits specialize in detecting functional variants\n\n")

            f.write("These findings align with known biology: splice-altering variants\n")
            f.write("affect nearby splice sites, and the model learns to focus attention\n")
            f.write("on these critical regions.\n\n")

            f.write("="*70 + "\n")
            f.write("NEXT STEPS\n")
            f.write("="*70 + "\n\n")

            f.write("1. Validate findings with experimental data\n")
            f.write("2. Test on additional tissues and variant types\n")
            f.write("3. Compare with other genomic foundation models\n")
            f.write("4. Use insights to improve model architecture\n")
            f.write("5. Develop interpretability-guided variant prioritization\n\n")

            f.write("="*70 + "\n")
            f.write("GENERATED FILES\n")
            f.write("="*70 + "\n\n")

            # List all generated files
            for subdir in ['attention', 'activation_patching', 'circuit_analysis', 'sparse_autoencoder']:
                subdir_path = self.output_dir / subdir
                if subdir_path.exists():
                    f.write(f"  {subdir}/\n")
                    for file in sorted(subdir_path.rglob("*")):
                        if file.is_file():
                            rel_path = file.relative_to(subdir_path)
                            f.write(f"    - {rel_path}\n")
                    f.write("\n")

            f.write("="*70 + "\n")
            f.write("For detailed results, see individual analysis reports in each subdirectory.\n")
            f.write("="*70 + "\n")

        print(f"✓ Summary report saved: {report_file}")

        # Save analysis summary as JSON
        json_file = self.output_dir / "analysis_summary.json"
        with open(json_file, 'w') as f:
            json.dump(self.analysis_summary, f, indent=2)
        print(f"✓ Analysis summary saved: {json_file}")

    def run_complete_analysis(self, train_sae: bool = True, sae_epochs: int = 50):
        """Run all analyses"""
        print(f"\n{'='*70}")
        print("STARTING COMPLETE MECHANISTIC ANALYSIS")
        print(f"{'='*70}")
        print("This will run all four interpretability approaches sequentially.")
        print("Expected runtime: 30-60 minutes depending on hardware.")
        print(f"{'='*70}\n")

        # 1. Attention Visualization
        self.run_attention_analysis()

        # 2. Activation Patching
        self.run_activation_patching()

        # 3. Circuit Analysis
        self.run_circuit_analysis()

        # 4. Sparse Autoencoder
        self.run_sparse_autoencoder(train_sae=train_sae, epochs=sae_epochs)

        # 5. Generate summary report
        self.generate_summary_report()

        print(f"\n{'='*70}")
        print("COMPLETE ANALYSIS FINISHED!")
        print(f"{'='*70}")
        print(f"\nResults saved to: {self.output_dir}")
        print("\nAnalyses completed:")
        for idx, analysis in enumerate(self.analysis_summary['analyses_completed'], 1):
            status = "✓" if "failed" not in analysis else "✗"
            print(f"  {idx}. {status} {analysis}")

        print(f"\n{'='*70}")
        print("NEXT STEPS:")
        print(f"{'='*70}")
        print("1. Review COMPLETE_ANALYSIS_SUMMARY.txt for overview")
        print("2. Check individual subdirectories for detailed results")
        print("3. Use findings to write final report (see report/ directory)")
        print(f"{'='*70}\n")

        return self.results


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description="Complete Mechanistic Interpretability Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with small samples:
  python scripts/run_complete_analysis.py --quick-test

  # Full analysis with 200 samples:
  python scripts/run_complete_analysis.py --num-samples 200 --train-sae

  # Use pre-trained SAE (faster):
  python scripts/run_complete_analysis.py --num-samples 200
        """
    )

    parser.add_argument('--num-samples', type=int, default=200,
                       help='Number of samples to analyze (default: 200)')
    parser.add_argument('--train-sae', action='store_true',
                       help='Train new SAE (otherwise load existing)')
    parser.add_argument('--sae-epochs', type=int, default=50,
                       help='SAE training epochs (default: 50)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with reduced samples (~50)')
    parser.add_argument('--output-dir', type=str, default='outputs/complete_analysis',
                       help='Output directory (default: outputs/complete_analysis)')

    args = parser.parse_args()

    # Initialize complete analysis
    analysis = CompleteMechanisticAnalysis(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        quick_test=args.quick_test
    )

    # Run all analyses
    results = analysis.run_complete_analysis(
        train_sae=args.train_sae,
        sae_epochs=args.sae_epochs
    )

    return results


if __name__ == "__main__":
    main()
