"""Explore and analyze the training data from genomic-FM."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import sys
from typing import Dict, List, Tuple

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_sqtl_data(data_path: str) -> Dict:
    """Load sQTL data from genomic-FM.

    Args:
        data_path: Path to data directory (e.g., genomic-FM/root/data)

    Returns:
        Dictionary with loaded data arrays
    """
    data_path = Path(data_path)

    # Try to find the sQTL data files
    sqtl_dir = data_path / "npy_output_delta_sqtl_pval_dnabert2"

    if not sqtl_dir.exists():
        print("No sQTL data directory found")
        return None

    data_dict = {}

    # Try to load sequence and label files
    for file_type in ['seq1', 'y', 'annot']:
        pattern = f"sqtl_pval_dnabert2_{file_type}_*.npy"
        files = sorted(sqtl_dir.glob(pattern))

        if files:
            print(f"\nLoading {file_type} files...")
            arrays = []
            for f in files:
                try:
                    # Try loading without pickle first (for standard numpy arrays)
                    arr = np.load(f)
                    arrays.append(arr)
                    print(f"  ✓ {f.name}: shape {arr.shape}, dtype {arr.dtype}")
                except ValueError:
                    # If that fails, try with allow_pickle=True
                    try:
                        arr = np.load(f, allow_pickle=True, encoding='bytes')
                        arrays.append(arr)
                        print(f"  ✓ {f.name}: shape {arr.shape}, dtype {arr.dtype} (pickled)")
                    except Exception as e2:
                        print(f"  ✗ {f.name}: Could not load - {type(e2).__name__}")
                except Exception as e:
                    print(f"  ✗ {f.name}: Could not load - {type(e).__name__}")

            if arrays:
                # Concatenate all chunks
                try:
                    data_dict[file_type] = np.concatenate(arrays) if len(arrays) > 1 else arrays[0]
                    print(f"  Combined {file_type}: shape {data_dict[file_type].shape}")
                except Exception as e:
                    print(f"  Could not concatenate {file_type} arrays: {e}")

    return data_dict if data_dict else None


def analyze_sequence_data(sequences: List[str], labels: List[int] = None) -> Dict:
    """Analyze DNA sequences.

    Args:
        sequences: List of DNA sequences
        labels: Optional labels for sequences

    Returns:
        Dictionary with analysis results
    """
    results = {}

    # Sequence length distribution
    lengths = [len(seq) for seq in sequences]
    results['lengths'] = {
        'mean': np.mean(lengths),
        'std': np.std(lengths),
        'min': np.min(lengths),
        'max': np.max(lengths),
        'distribution': lengths
    }

    # Nucleotide composition
    all_nucs = ''.join(sequences)
    nuc_counts = Counter(all_nucs)
    total = sum(nuc_counts.values())
    results['nucleotide_freq'] = {nuc: count/total for nuc, count in nuc_counts.items()}

    # GC content
    gc_content = [(seq.count('G') + seq.count('C')) / len(seq) for seq in sequences]
    results['gc_content'] = {
        'mean': np.mean(gc_content),
        'std': np.std(gc_content),
        'distribution': gc_content
    }

    # Label distribution if provided
    if labels is not None:
        label_counts = Counter(labels)
        results['label_distribution'] = dict(label_counts)
        results['class_balance'] = {
            label: count/len(labels)
            for label, count in label_counts.items()
        }

    return results


def plot_sequence_analysis(results: Dict, save_path: str = None):
    """Create comprehensive plots for sequence analysis.

    Args:
        results: Results from analyze_sequence_data
        save_path: Optional path to save plots
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Sequence Data Analysis', fontsize=16, fontweight='bold')

    # 1. Sequence length distribution
    ax = axes[0, 0]
    lengths = results['lengths']['distribution']
    ax.hist(lengths, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(results['lengths']['mean'], color='red', linestyle='--',
               label=f"Mean: {results['lengths']['mean']:.0f}")
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Frequency')
    ax.set_title('Sequence Length Distribution')
    ax.legend()

    # 2. Nucleotide frequency
    ax = axes[0, 1]
    nucs = list(results['nucleotide_freq'].keys())
    freqs = list(results['nucleotide_freq'].values())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(nucs)]
    ax.bar(nucs, freqs, color=colors, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Nucleotide')
    ax.set_ylabel('Frequency')
    ax.set_title('Nucleotide Composition')
    ax.set_ylim([0, max(freqs) * 1.2])
    for i, (nuc, freq) in enumerate(zip(nucs, freqs)):
        ax.text(i, freq + 0.01, f'{freq:.3f}', ha='center', va='bottom')

    # 3. GC content distribution
    ax = axes[0, 2]
    gc_content = results['gc_content']['distribution']
    ax.hist(gc_content, bins=50, edgecolor='black', alpha=0.7, color='green')
    ax.axvline(results['gc_content']['mean'], color='red', linestyle='--',
               label=f"Mean: {results['gc_content']['mean']:.3f}")
    ax.set_xlabel('GC Content')
    ax.set_ylabel('Frequency')
    ax.set_title('GC Content Distribution')
    ax.legend()

    # 4. Label distribution (if available)
    if 'label_distribution' in results:
        ax = axes[1, 0]
        labels = list(results['label_distribution'].keys())
        counts = list(results['label_distribution'].values())
        ax.bar([str(l) for l in labels], counts, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Label')
        ax.set_ylabel('Count')
        ax.set_title('Label Distribution')
        for i, count in enumerate(counts):
            ax.text(i, count + max(counts)*0.02, str(count), ha='center', va='bottom')

    # 5. Class balance (if available)
    if 'class_balance' in results:
        ax = axes[1, 1]
        labels = list(results['class_balance'].keys())
        balance = list(results['class_balance'].values())
        ax.pie(balance, labels=[str(l) for l in labels], autopct='%1.1f%%',
               startangle=90, colors=['#ff9999', '#66b3ff'])
        ax.set_title('Class Balance')

    # 6. Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = f"""
    Summary Statistics:

    Sequence Length:
      Mean: {results['lengths']['mean']:.2f}
      Std: {results['lengths']['std']:.2f}
      Range: [{results['lengths']['min']}, {results['lengths']['max']}]

    GC Content:
      Mean: {results['gc_content']['mean']:.3f}
      Std: {results['gc_content']['std']:.3f}

    Nucleotide Frequencies:
    """
    for nuc, freq in results['nucleotide_freq'].items():
        summary_text += f"      {nuc}: {freq:.4f}\n"

    ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()
    return fig


def analyze_variant_effects(data_path: str, output_dir: Path):
    """Analyze variant effect prediction data.

    Args:
        data_path: Path to genomic-FM data directory
        output_dir: Directory to save visualizations
    """
    data_path = Path(data_path)

    # Look for variant data
    print(f"\nSearching for variant data in: {data_path}")

    # Check for different file patterns
    patterns = ['*.vcf', '*.tsv', '*.csv']

    results = {}
    for pattern in patterns:
        files = list(data_path.glob(pattern))
        if files:
            print(f"\nFound files matching {pattern}:")
            for f in files[:10]:  # Limit to first 10
                size_mb = f.stat().st_size / 1024 / 1024
                print(f"  - {f.name} ({size_mb:.2f} MB)")
                results[f.name] = size_mb

    # Save file inventory as text report
    report_path = output_dir / 'data_files_inventory.txt'
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("GENOMIC DATA FILES INVENTORY\n")
        f.write("="*60 + "\n\n")

        for pattern in patterns:
            files = list(data_path.glob(pattern))
            if files:
                f.write(f"\n{pattern} files:\n")
                for file in files[:10]:
                    size_mb = file.stat().st_size / 1024 / 1024
                    f.write(f"  - {file.name} ({size_mb:.2f} MB)\n")

        f.write(f"\n{'='*60}\n")
        f.write(f"Total files inventoried: {len(results)}\n")
        f.write(f"Total size: {sum(results.values()):.2f} MB\n")

    print(f"\n  Saved inventory: {report_path}")

    # Try to load and analyze GWAS data
    gwas_files = list(data_path.glob('*gwas*.tsv'))
    if gwas_files:
        print(f"\n{'='*60}")
        print("ANALYZING GWAS CATALOG DATA")
        print('='*60)
        try:
            gwas_df = pd.read_csv(gwas_files[0], sep='\t', nrows=1000, low_memory=False)
            print(f"Loaded {len(gwas_df)} GWAS associations (sample)")
            print(f"Columns: {', '.join(gwas_df.columns[:10].tolist())}...")

            # Save GWAS summary
            gwas_report_path = output_dir / 'gwas_summary.txt'
            with open(gwas_report_path, 'w') as f:
                f.write("="*60 + "\n")
                f.write("GWAS CATALOG SUMMARY (sample of 1000 records)\n")
                f.write("="*60 + "\n\n")
                f.write(f"Source file: {gwas_files[0].name}\n")
                f.write(f"Records analyzed: {len(gwas_df)}\n")
                f.write(f"Columns: {len(gwas_df.columns)}\n\n")
                f.write("Column names:\n")
                for i, col in enumerate(gwas_df.columns, 1):
                    f.write(f"  {i}. {col}\n")

            print(f"  Saved GWAS summary: {gwas_report_path}")

            # Count by chromosome if available
            if 'CHR_ID' in gwas_df.columns or 'CHROMOSOME' in gwas_df.columns:
                chr_col = 'CHR_ID' if 'CHR_ID' in gwas_df.columns else 'CHROMOSOME'
                print(f"\nAssociations by chromosome:")
                chr_counts = gwas_df[chr_col].value_counts().head(10)
                print(chr_counts)

                # Create visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                chr_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
                ax.set_xlabel('Chromosome')
                ax.set_ylabel('Number of Associations')
                ax.set_title('GWAS Associations by Chromosome (Top 10)')
                plt.xticks(rotation=45)
                plt.tight_layout()

                chart_path = output_dir / 'gwas_chromosome_distribution.png'
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                print(f"  Saved chromosome chart: {chart_path}")
                plt.close()

        except Exception as e:
            print(f"Could not analyze GWAS file: {e}")

    # Try to load ClinVar data
    clinvar_files = list(data_path.glob('*clinvar*.vcf'))
    if clinvar_files:
        print(f"\n{'='*60}")
        print("CLINVAR DATA FOUND")
        print('='*60)

        clinvar_report_path = output_dir / 'clinvar_summary.txt'
        with open(clinvar_report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("CLINVAR DATA SUMMARY\n")
            f.write("="*60 + "\n\n")
            for file in clinvar_files:
                size_mb = file.stat().st_size / 1024 / 1024
                f.write(f"File: {file.name}\n")
                f.write(f"Size: {size_mb:.2f} MB\n")
                print(f"  - {file.name} ({size_mb:.2f} MB)")

        print(f"  Saved ClinVar summary: {clinvar_report_path}")


def create_data_exploration_report(data_dir: str, output_dir: str = "./reports"):
    """Create comprehensive data exploration report.

    Args:
        data_dir: Path to genomic-FM data directory
        output_dir: Directory to save report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("="*60)
    print("GENOMIC DATA EXPLORATION REPORT")
    print("="*60)

    # Load and analyze sQTL data
    print("\n" + "="*60)
    print("LOADING SQTL TRAINING DATA")
    print("="*60)

    sqtl_data = load_sqtl_data(data_dir)

    if sqtl_data is not None:
        print(f"\n✓ Data loaded successfully!")
        print(f"\nData summary:")
        for key, arr in sqtl_data.items():
            print(f"  {key:10s}: shape {arr.shape}, dtype {arr.dtype}")
            if key == 'y':
                print(f"             Labels - min: {arr.min():.3f}, max: {arr.max():.3f}, mean: {arr.mean():.3f}")

        # Create visualization if sequences are available
        if 'seq1' in sqtl_data:
            print(f"\n  Sequence data: {sqtl_data['seq1'].shape[0]} sequences")
            print(f"  Sequence length: {sqtl_data['seq1'].shape[1]} tokens")

            # Plot label distribution if available
            if 'y' in sqtl_data:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(sqtl_data['y'], bins=50, edgecolor='black', alpha=0.7)
                ax.set_xlabel('Label Value')
                ax.set_ylabel('Frequency')
                ax.set_title('sQTL Label Distribution')
                plt.savefig(output_dir / 'sqtl_label_distribution.png', dpi=300, bbox_inches='tight')
                print(f"\n  Saved label distribution plot: {output_dir / 'sqtl_label_distribution.png'}")
                plt.close()
    else:
        print("\n⚠ Could not load sQTL training data (numpy files may be from incompatible version)")
        print("  This is non-critical - continuing with variant database analysis")

    # Analyze variant databases
    analyze_variant_effects(data_dir, output_dir)

    print("\n" + "="*60)
    print("✓ Report generation complete!")
    print(f"  Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Explore genomic training data")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../genomic-FM/root/data",
        help="Path to genomic-FM data directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./reports",
        help="Output directory for plots and reports"
    )

    args = parser.parse_args()

    create_data_exploration_report(args.data_dir, args.output_dir)
