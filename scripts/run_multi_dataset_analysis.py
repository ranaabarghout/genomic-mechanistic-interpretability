#!/usr/bin/env python3
"""
Universal mechanistic interpretability analysis for any genomic dataset.
Supports all datasets from genomic-FM repository.
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src and scripts directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from data.generic_data_loader import get_data_loader
from run_sqtl_attention_analysis import sQTLAttentionAnalyzer
from run_activation_patching import sQTLActivationPatcher
from run_circuit_analysis import CircuitAnalyzer
from run_sparse_autoencoder import sQTLSAEAnalyzer


class UniversalMechanisticAnalysis:
    """Universal mechanistic interpretability analysis for any dataset."""

    # Dataset configurations
    DATASET_CONFIGS = {
        'sqtl': {
            'description': 'Splicing QTL variants (splice-altering)',
            'default_samples': 200,
            'tissue_filter': 'Whole_Blood',
            'binary': True
        },
        'eqtl': {
            'description': 'Expression QTL variants (expression-altering)',
            'default_samples': 200,
            'tissue_filter': 'Whole_Blood',
            'binary': True
        },
        'clinvar': {
            'description': 'Clinical variants (pathogenic vs benign)',
            'default_samples': 200,
            'binary': True
        },
        'gwas': {
            'description': 'GWAS variants (trait-associated)',
            'default_samples': 200,
            'binary': True
        },
        'mave': {
            'description': 'MAVE variants (experimental effect scores)',
            'default_samples': 200,
            'binary': False,  # Typically regression
            'classification': True  # Can convert to classification
        },
        'geneko': {
            'description': 'Gene knockout effects',
            'default_samples': 200,
            'binary': True
        },
        'cellpassport': {
            'description': 'Cell line drug response',
            'default_samples': 200,
            'binary': False
        },
        'oligogenic': {
            'description': 'Oligogenic disease variants',
            'default_samples': 200,
            'binary': True
        }
    }

    def __init__(
        self,
        dataset_name: str,
        num_samples: int = None,
        output_dir: Path = None,
        quick_test: bool = False,
        train_sae: bool = True,
        sae_epochs: int = 50
    ):
        """
        Initialize universal analysis.

        Args:
            dataset_name: Name of dataset (sqtl, eqtl, clinvar, gwas, mave, etc.)
            num_samples: Number of samples to analyze
            output_dir: Output directory
            quick_test: Quick test mode
            train_sae: Whether to train SAE
            sae_epochs: SAE training epochs
        """
        self.dataset_name = dataset_name.lower()

        if self.dataset_name not in self.DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}. "
                           f"Supported: {', '.join(self.DATASET_CONFIGS.keys())}")

        self.config = self.DATASET_CONFIGS[self.dataset_name]

        # Set defaults
        if num_samples is None:
            num_samples = 50 if quick_test else self.config['default_samples']

        self.num_samples = num_samples
        self.quick_test = quick_test
        self.train_sae = train_sae
        self.sae_epochs = 20 if quick_test else sae_epochs

        # Setup output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f"outputs/{dataset_name}_analysis/analysis_{timestamp}")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Results tracking
        self.results = {
            'attention': None,
            'activation_patching': None,
            'circuit_analysis': None,
            'sparse_autoencoder': None
        }

        print("="*70)
        print(f"UNIVERSAL MECHANISTIC INTERPRETABILITY ANALYSIS")
        print("="*70)
        print(f"Dataset: {self.dataset_name.upper()}")
        print(f"Description: {self.config['description']}")
        print(f"Output directory: {self.output_dir}")
        print(f"Samples: {self.num_samples}")
        print(f"Quick test: {self.quick_test}")
        print("="*70)
        print()

    def load_data(self):
        """Load dataset using appropriate data loader."""
        print("="*70)
        print(f"LOADING {self.dataset_name.upper()} DATA")
        print("="*70)
        print()

        # Get appropriate data loader
        loader_kwargs = {'num_records': self.num_samples, 'seq_length': 1024}

        # Add dataset-specific arguments
        if self.dataset_name in ['sqtl', 'eqtl'] and 'tissue_filter' in self.config:
            loader_kwargs['tissue_filter'] = self.config['tissue_filter']
        elif self.dataset_name == 'clinvar':
            loader_kwargs['binary_classification'] = True
            loader_kwargs['include_vus'] = False
        elif self.dataset_name == 'mave':
            loader_kwargs['classification'] = self.config.get('classification', False)

        self.data_loader = get_data_loader(self.dataset_name, **loader_kwargs)
        self.samples = self.data_loader.load_data()

        self.data_loader.print_statistics()

        return self.samples

    def run_attention_analysis(self):
        """Run attention visualization analysis."""
        print("\n" + "="*70)
        print("ANALYSIS 1/4: ATTENTION VISUALIZATION")
        print("="*70)
        print()

        try:
            output_subdir = self.output_dir / "attention"
            output_subdir.mkdir(exist_ok=True)

            analyzer = sQTLAttentionAnalyzer(
                output_dir=str(output_subdir),
                model_name="zhihan1996/DNABERT-2-117M"
            )

            # Load data into analyzer format
            analyzer.samples = self.samples
            analyzer.significant = [s for s in self.samples if self.data_loader.get_label_name(s.label) in ['significant', 'pathogenic', 'gain_of_function']]
            analyzer.not_significant = [s for s in self.samples if self.data_loader.get_label_name(s.label) in ['not_significant', 'benign', 'neutral', 'loss_of_function']]

            # Run full analysis
            results = analyzer.run_full_analysis()

            self.results['attention'] = True
            print("\n✓ Attention analysis complete!")

        except Exception as e:
            print(f"\n✗ Attention analysis failed: {e}")
            import traceback
            traceback.print_exc()
            self.results['attention'] = False

    def run_activation_patching_analysis(self):
        """Run activation patching analysis."""
        print("\n" + "="*70)
        print("ANALYSIS 2/4: ACTIVATION PATCHING")
        print("="*70)
        print()

        try:
            output_subdir = self.output_dir / "activation_patching"
            output_subdir.mkdir(exist_ok=True)

            patcher = sQTLActivationPatcher(
                output_dir=str(output_subdir),
                model_name="zhihan1996/DNABERT-2-117M"
            )

            # Load data
            patcher.samples = self.samples
            patcher.significant = [s for s in self.samples if self.data_loader.get_label_name(s.label) in ['significant', 'pathogenic', 'gain_of_function']]
            patcher.not_significant = [s for s in self.samples if self.data_loader.get_label_name(s.label) in ['not_significant', 'benign', 'neutral', 'loss_of_function']]

            # Run analysis
            results = patcher.run_full_analysis()

            self.results['activation_patching'] = True
            print("\n✓ Activation patching complete!")

        except Exception as e:
            print(f"\n✗ Activation patching failed: {e}")
            import traceback
            traceback.print_exc()
            self.results['activation_patching'] = False

    def run_circuit_analysis(self):
        """Run circuit analysis."""
        print("\n" + "="*70)
        print("ANALYSIS 3/4: CIRCUIT ANALYSIS & ABLATION")
        print("="*70)
        print()

        try:
            output_subdir = self.output_dir / "circuit_analysis"
            output_subdir.mkdir(exist_ok=True)

            analyzer = CircuitAnalyzer(
                output_dir=str(output_subdir),
                model_name="zhihan1996/DNABERT-2-117M"
            )

            # Load data
            analyzer.samples = self.samples
            analyzer.significant = [s for s in self.samples if self.data_loader.get_label_name(s.label) in ['significant', 'pathogenic', 'gain_of_function']]
            analyzer.not_significant = [s for s in self.samples if self.data_loader.get_label_name(s.label) in ['not_significant', 'benign', 'neutral', 'loss_of_function']]

            # Run analysis (no parameters needed)
            results = analyzer.run_full_analysis()

            self.results['circuit_analysis'] = True
            print("\n✓ Circuit analysis complete!")

        except Exception as e:
            print(f"\n✗ Circuit analysis failed: {e}")
            import traceback
            traceback.print_exc()
            self.results['circuit_analysis'] = False

    def run_sae_analysis(self):
        """Run sparse autoencoder analysis."""
        if not self.train_sae:
            print("\n✗ Sparse autoencoder training skipped (--train-sae not set)")
            self.results['sparse_autoencoder'] = False
            return

        print("\n" + "="*70)
        print("ANALYSIS 4/4: SPARSE AUTOENCODER")
        print("="*70)
        print()

        try:
            output_subdir = self.output_dir / "sparse_autoencoder"
            output_subdir.mkdir(exist_ok=True)

            analyzer = sQTLSAEAnalyzer(
                output_dir=str(output_subdir),
                model_name="zhihan1996/DNABERT-2-117M"
            )

            # Load data
            analyzer.samples = self.samples

            # Run analysis
            results = analyzer.run_full_analysis(train_sae=True, epochs=self.sae_epochs)

            self.results['sparse_autoencoder'] = True
            print("\n✓ Sparse autoencoder analysis complete!")

        except Exception as e:
            print(f"\n✗ Sparse autoencoder analysis failed: {e}")
            import traceback
            traceback.print_exc()
            self.results['sparse_autoencoder'] = False

    def generate_summary(self):
        """Generate summary report."""
        print("\n" + "="*70)
        print("GENERATING SUMMARY REPORT")
        print("="*70)

        summary_path = self.output_dir / f"{self.dataset_name.upper()}_ANALYSIS_SUMMARY.txt"

        with open(summary_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write(f"MECHANISTIC INTERPRETABILITY ANALYSIS: {self.dataset_name.upper()}\n")
            f.write(f"{self.config['description']}\n")
            f.write("="*70 + "\n\n")

            f.write(f"Analysis Timestamp: {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
            f.write(f"Samples Analyzed: {self.num_samples}\n")
            f.write(f"Quick Test Mode: {self.quick_test}\n\n")

            f.write("="*70 + "\n")
            f.write("ANALYSES COMPLETED\n")
            f.write("="*70 + "\n\n")

            for i, (analysis, status) in enumerate(self.results.items(), 1):
                status_symbol = "✓" if status else "✗"
                f.write(f"  {i}. {status_symbol} {analysis.upper().replace('_', ' ')}\n")

            f.write("\n" + "="*70 + "\n")
            f.write(f"Results saved to: {self.output_dir}\n")
            f.write("="*70 + "\n")

        print(f"\n✓ Summary report saved: {summary_path}")

        return summary_path

    def run(self):
        """Run complete analysis pipeline."""
        print("\n" + "="*70)
        print(f"STARTING {self.dataset_name.upper()} ANALYSIS PIPELINE")
        print("="*70)
        print()

        # Load data
        self.load_data()

        # Run analyses
        self.run_attention_analysis()
        self.run_activation_patching_analysis()
        self.run_circuit_analysis()
        self.run_sae_analysis()

        # Generate summary
        summary_path = self.generate_summary()

        # Final report
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        print(f"\nDataset: {self.dataset_name.upper()}")
        print(f"Results: {self.output_dir}")
        print(f"\nAnalyses completed:")
        for analysis, status in self.results.items():
            status_symbol = "✓" if status else "✗"
            print(f"  {status_symbol} {analysis.replace('_', ' ').title()}")

        print("\n" + "="*70)
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Universal mechanistic interpretability analysis for genomic datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # eQTL analysis
  python scripts/run_multi_dataset_analysis.py --dataset eqtl --num-samples 200 --train-sae

  # ClinVar pathogenicity
  python scripts/run_multi_dataset_analysis.py --dataset clinvar --num-samples 200

  # GWAS traits
  python scripts/run_multi_dataset_analysis.py --dataset gwas --num-samples 200

  # Quick test on any dataset
  python scripts/run_multi_dataset_analysis.py --dataset mave --quick-test

Supported datasets:
  sqtl       - Splicing QTL variants
  eqtl       - Expression QTL variants
  clinvar    - Clinical variants (pathogenic vs benign)
  gwas       - GWAS trait-associated variants
  mave       - MAVE experimental effect scores
  geneko     - Gene knockout effects
  cellpassport - Cell line drug response
  oligogenic - Oligogenic disease variants
        """
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=list(UniversalMechanisticAnalysis.DATASET_CONFIGS.keys()),
        help='Dataset to analyze'
    )

    parser.add_argument(
        '--num-samples',
        type=int,
        default=None,
        help='Number of samples to analyze (default: dataset-specific)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: outputs/<dataset>_analysis/analysis_<timestamp>)'
    )

    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test mode with reduced samples and epochs'
    )

    parser.add_argument(
        '--train-sae',
        action='store_true',
        help='Train sparse autoencoder (takes extra time)'
    )

    parser.add_argument(
        '--sae-epochs',
        type=int,
        default=50,
        help='Number of SAE training epochs (default: 50)'
    )

    args = parser.parse_args()

    # Run analysis
    analysis = UniversalMechanisticAnalysis(
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        quick_test=args.quick_test,
        train_sae=args.train_sae,
        sae_epochs=args.sae_epochs
    )

    analysis.run()


if __name__ == "__main__":
    main()
