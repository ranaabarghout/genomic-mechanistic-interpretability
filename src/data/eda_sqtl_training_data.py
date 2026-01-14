"""
Exploratory Data Analysis for sQTL Training Data
=================================================
This script analyzes the training data used for fine-tuning DNABERT-2 model
on the sqtl_pval_dnabert2 dataset.

Dataset: sQTL (Splicing Quantitative Trait Loci) from GTEx v8
Task: Binary classification (significant vs not_significant)
Target: p_val (pval_nominal < pval_nominal_threshold)
Sequence length: 1024bp
Organism: Whole_Blood
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class sQTLDataExplorer:
    """Comprehensive EDA for sQTL training data"""

    def __init__(self, data_path: str):
        """
        Initialize data explorer

        Args:
            data_path: Path to npy_output_delta_sqtl_pval_dnabert2 directory
        """
        self.data_path = Path(data_path)
        self.output_dir = Path("outputs/sqtl_eda")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load metadata
        self.x_class = self._load_yaml("sqtl_pval_dnabert2_x_class.yaml")
        self.y_class = self._load_yaml("sqtl_pval_dnabert2_y_class.yaml")

        # Initialize data containers
        self.sequences = []
        self.labels = []
        self.annotations = []

    def _load_yaml(self, filename: str):
        """Load YAML file"""
        with open(self.data_path / filename, 'r') as f:
            return yaml.safe_load(f)

    def load_data(self):
        """Load all numpy data chunks (stored as raw binary data)"""
        print("Loading data chunks...")
        print("Note: Files are stored as raw binary (not standard numpy format)")

        # Data is split into chunks: 0, 2500, 5000
        chunks = [0, 2500, 5000]
        # PCA components used during saving (from config)
        pca_components = 16
        # Sequence length from config
        seq_length = 1024

        for chunk_id in chunks:
            try:
                # Load labels first to know how many samples
                y_file = self.data_path / f"sqtl_pval_dnabert2_y_{chunk_id}.npy"
                with open(y_file, 'rb') as f:
                    buffer = f.read()
                    y_data = np.frombuffer(buffer, dtype=np.int64)
                    num_samples = len(y_data)
                    self.labels.append(y_data)
                    print(f"  Loaded y_{chunk_id}: shape {y_data.shape}, dtype {y_data.dtype}")

                # Load sequences (seq1) - stored as float32 after PCA
                # Shape: (num_samples, seq_length, pca_components) but flattened
                seq_file = self.data_path / f"sqtl_pval_dnabert2_seq1_{chunk_id}.npy"
                with open(seq_file, 'rb') as f:
                    buffer = f.read()
                    # Load as flat array and reshape to (num_samples, seq_length, pca_components)
                    seq_data = np.frombuffer(buffer, dtype=np.float32).reshape(
                        num_samples, seq_length, pca_components
                    )
                    self.sequences.append(seq_data)
                    print(f"  Loaded seq1_{chunk_id}: shape {seq_data.shape}, dtype {seq_data.dtype}")

                # Load annotations (annot) - stored as int64
                annot_file = self.data_path / f"sqtl_pval_dnabert2_annot_{chunk_id}.npy"
                with open(annot_file, 'rb') as f:
                    buffer = f.read()
                    annot_data = np.frombuffer(buffer, dtype=np.int64)
                    self.annotations.append(annot_data)
                    print(f"  Loaded annot_{chunk_id}: shape {annot_data.shape}, dtype {annot_data.dtype}")

            except Exception as e:
                print(f"  Error loading chunk {chunk_id}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Concatenate all chunks
        if self.sequences:
            self.sequences = np.concatenate(self.sequences, axis=0)
            print(f"\nTotal sequences: {len(self.sequences)}")

        if self.labels:
            self.labels = np.concatenate(self.labels, axis=0)
            print(f"Total labels: {len(self.labels)}")

        if self.annotations:
            self.annotations = np.concatenate(self.annotations, axis=0)
            print(f"Total annotations: {len(self.annotations)}")

    def analyze_dataset_overview(self):
        """Generate dataset overview statistics"""
        print("\n" + "="*70)
        print("DATASET OVERVIEW")
        print("="*70)

        report = []
        report.append("="*70)
        report.append("sQTL Training Data - Exploratory Data Analysis")
        report.append("="*70)
        report.append("")

        # Basic stats
        report.append("Dataset Information:")
        report.append(f"  Total samples: {len(self.sequences):,}")
        report.append(f"  Sequence shape: {self.sequences.shape}")
        report.append(f"  Labels shape: {self.labels.shape}")
        report.append(f"  Annotations shape: {self.annotations.shape}")
        report.append("")

        # Label distribution
        report.append("Label Classes:")
        for label, idx in self.y_class.items():
            count = np.sum(self.labels == idx)
            percentage = (count / len(self.labels)) * 100
            report.append(f"  {label} (class {idx}): {count:,} samples ({percentage:.2f}%)")
        report.append("")

        # Organism information
        report.append("Organism Classes:")
        for organism, idx in self.x_class.items():
            report.append(f"  {organism}: class {idx}")
        report.append("")

        # Data characteristics
        report.append("Data Characteristics:")
        report.append(f"  Task: Binary classification (sQTL significance)")
        report.append(f"  Target: p_val (pval_nominal vs pval_nominal_threshold)")
        report.append(f"  Sequence type: Reference and Alternate genomic sequences")
        report.append(f"  Organism: Whole_Blood (GTEx v8)")
        report.append("")

        # Print and save
        report_text = "\n".join(report)
        print(report_text)

        with open(self.output_dir / "dataset_overview.txt", "w") as f:
            f.write(report_text)

        print(f"\nSaved: {self.output_dir / 'dataset_overview.txt'}")

    def analyze_class_distribution(self):
        """Analyze and visualize class distribution"""
        print("\n" + "="*70)
        print("CLASS DISTRIBUTION ANALYSIS")
        print("="*70)

        # Count classes
        unique, counts = np.unique(self.labels, return_counts=True)

        # Create reverse mapping
        class_names = {v: k for k, v in self.y_class.items()}

        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Bar plot
        ax = axes[0]
        labels_text = [class_names[u] for u in unique]
        colors = ['#2ecc71' if 'significant' in name else '#e74c3c' for name in labels_text]
        bars = ax.bar(labels_text, counts, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count:,}\n({count/len(self.labels)*100:.1f}%)',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Pie chart
        ax = axes[1]
        ax.pie(counts, labels=labels_text, autopct='%1.1f%%',
               colors=colors, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax.set_title('Class Proportion', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / "class_distribution.png", dpi=300, bbox_inches='tight')
        print(f"Saved: {self.output_dir / 'class_distribution.png'}")
        plt.close()

        # Calculate class imbalance ratio
        imbalance_ratio = max(counts) / min(counts)
        print(f"\nClass Imbalance Ratio: {imbalance_ratio:.2f}:1")

        if imbalance_ratio > 3:
            print("⚠️  WARNING: Significant class imbalance detected!")
            print("   Consider using class weights or resampling techniques.")

    def analyze_sequence_characteristics(self):
        """Analyze sequence-level characteristics (PCA-transformed features)"""
        print("\n" + "="*70)
        print("SEQUENCE CHARACTERISTICS ANALYSIS")
        print("="*70)

        print(f"\nNote: Sequences are PCA-transformed ({self.sequences.shape[2]} components)")
        print(f"Shape: {self.sequences.shape} (samples, seq_length, pca_components)")

        # The sequences are PCA-transformed embeddings, not raw DNA
        # Shape: (num_samples, seq_length=1024, pca_components=16)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Distribution of PCA components across all positions (sample)
        ax = axes[0, 0]
        sample_size = min(500, len(self.sequences))
        sample_indices = np.random.choice(len(self.sequences), sample_size, replace=False)
        sample_data = self.sequences[sample_indices]  # (sample_size, 1024, 16)

        # Flatten across sequence positions to get distribution of each PC
        # Plot first 4 PCA components
        for i in range(min(4, sample_data.shape[2])):
            pc_values = sample_data[:, :, i].flatten()  # All positions for this PC
            ax.hist(pc_values, bins=50, alpha=0.5, label=f'PC{i+1}')
        ax.set_xlabel('PCA Component Value', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title('Distribution of First 4 PCA Components', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # Average PCA values across sequence positions
        ax = axes[0, 1]
        # Average across samples and sequence positions for each PCA component
        pc_means = np.mean(self.sequences, axis=(0, 1))  # Mean across samples and positions
        pc_stds = np.std(self.sequences, axis=(0, 1))
        x_pos = np.arange(len(pc_means))
        ax.bar(x_pos, pc_means, yerr=pc_stds, capsize=5, alpha=0.7,
               color='#3498db', edgecolor='black')
        ax.set_xlabel('PCA Component', fontsize=11, fontweight='bold')
        ax.set_ylabel('Mean Value ± Std', fontsize=11, fontweight='bold')
        ax.set_title('Average PCA Component Values', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'PC{i+1}' for i in range(len(pc_means))], rotation=45)
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

        # Sequence representation magnitude per sample
        ax = axes[1, 0]
        # Calculate L2 norm across sequence_length and pca_components for each sample
        sequence_magnitudes = np.linalg.norm(self.sequences.reshape(len(self.sequences), -1), axis=1)
        ax.hist(sequence_magnitudes, bins=50, color='#9b59b6', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(sequence_magnitudes), color='red', linestyle='--',
                  linewidth=2, label=f'Mean: {np.mean(sequence_magnitudes):.1f}')
        ax.set_xlabel('Sequence Representation Magnitude', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title('Sequence Representation Magnitudes', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # Feature distribution by class
        ax = axes[1, 1]
        class_names = {v: k for k, v in self.y_class.items()}

        # Compare sequence magnitudes by class
        magnitudes_by_class = {}
        for class_idx in np.unique(self.labels):
            class_mask = self.labels == class_idx
            class_sequences = self.sequences[class_mask]
            class_magnitudes = np.linalg.norm(class_sequences.reshape(len(class_sequences), -1), axis=1)
            magnitudes_by_class[class_names[class_idx]] = class_magnitudes

        box_data = [magnitudes_by_class[label] for label in magnitudes_by_class.keys()]
        bp = ax.boxplot(box_data, labels=list(magnitudes_by_class.keys()),
                       patch_artist=True, showmeans=True)

        # Color boxes
        colors = ['#2ecc71' if 'significant' in label else '#e74c3c'
                 for label in magnitudes_by_class.keys()]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_ylabel('Sequence Magnitude', fontsize=11, fontweight='bold')
        ax.set_title('Sequence Magnitudes by Class', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "sequence_characteristics.png", dpi=300, bbox_inches='tight')
        print(f"Saved: {self.output_dir / 'sequence_characteristics.png'}")
        plt.close()

        # Print statistics
        print(f"\nPCA Feature Statistics:")
        print(f"  Sequence shape: {self.sequences.shape}")
        print(f"  Number of PCA components: {self.sequences.shape[2]}")
        print(f"  Sequence length: {self.sequences.shape[1]}")
        print(f"  Sequence magnitude: {np.mean(sequence_magnitudes):.2f} ± {np.std(sequence_magnitudes):.2f}")
        print(f"  PCA component means (averaged): {pc_means}")
        print(f"  PCA component stds: {pc_stds}")

        # Per-class statistics
        print(f"\nSequence Magnitude by Class:")
        for class_name, mags in magnitudes_by_class.items():
            print(f"  {class_name}: {np.mean(mags):.2f} ± {np.std(mags):.2f}")

    def analyze_data_quality(self):
        """Analyze data quality metrics"""
        print("\n" + "="*70)
        print("DATA QUALITY ANALYSIS")
        print("="*70)

        quality_report = []
        quality_report.append("Data Quality Assessment:")
        quality_report.append("")

        # Check for missing values
        missing_seq = np.sum(self.sequences == None) if self.sequences.dtype == object else 0
        missing_labels = np.sum(self.labels == None) if self.labels.dtype == object else 0

        quality_report.append(f"Missing Values:")
        quality_report.append(f"  Sequences: {missing_seq}")
        quality_report.append(f"  Labels: {missing_labels}")
        quality_report.append("")

        # Check data shapes consistency
        quality_report.append(f"Data Consistency:")
        quality_report.append(f"  Sequence count: {len(self.sequences):,}")
        quality_report.append(f"  Label count: {len(self.labels):,}")
        quality_report.append(f"  Match: {'✓' if len(self.sequences) == len(self.labels) else '✗ MISMATCH!'}")
        quality_report.append("")

        # Annotation info
        if len(self.annotations) > 0:
            quality_report.append(f"Annotations:")
            quality_report.append(f"  Total annotations: {len(self.annotations):,}")
            if isinstance(self.annotations[0], str):
                quality_report.append(f"  Sample annotation: {self.annotations[0]}")
        quality_report.append("")

        report_text = "\n".join(quality_report)
        print(report_text)

        with open(self.output_dir / "data_quality.txt", "w") as f:
            f.write(report_text)

        print(f"\nSaved: {self.output_dir / 'data_quality.txt'}")

    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*70)
        print("GENERATING COMPREHENSIVE SUMMARY REPORT")
        print("="*70)

        report = []
        report.append("="*80)
        report.append("COMPREHENSIVE EDA REPORT: sQTL Training Data")
        report.append("="*80)
        report.append("")
        report.append("Dataset: GTEx v8 Splicing Quantitative Trait Loci (sQTL)")
        report.append("Organism: Whole_Blood")
        report.append("Task: Binary Classification (sQTL significance)")
        report.append("Model: DNABERT-2-117M")
        report.append("Training: 100 epochs, classification head with output_size=2")
        report.append("")
        report.append("-"*80)
        report.append("1. DATASET OVERVIEW")
        report.append("-"*80)
        report.append(f"Total Samples: {len(self.sequences):,}")
        report.append(f"Sequence Shape: {self.sequences.shape}")
        report.append(f"Label Shape: {self.labels.shape}")
        report.append(f"Annotation Shape: {self.annotations.shape}")
        report.append("")

        report.append("-"*80)
        report.append("2. LABEL DISTRIBUTION")
        report.append("-"*80)
        class_names = {v: k for k, v in self.y_class.items()}
        for class_idx in np.unique(self.labels):
            count = np.sum(self.labels == class_idx)
            percentage = (count / len(self.labels)) * 100
            report.append(f"{class_names[class_idx]:20s}: {count:6,} samples ({percentage:5.2f}%)")

        # Calculate imbalance
        unique, counts = np.unique(self.labels, return_counts=True)
        imbalance_ratio = max(counts) / min(counts)
        report.append(f"\nClass Imbalance Ratio: {imbalance_ratio:.2f}:1")
        report.append("")

        report.append("-"*80)
        report.append("3. DATA SPLITS")
        report.append("-"*80)
        report.append("Data is stored in 3 chunks:")
        for chunk_id in [0, 2500, 5000]:
            report.append(f"  Chunk {chunk_id}: Files with suffix _{chunk_id}.npy")
        report.append("")

        report.append("-"*80)
        report.append("4. DATA CHARACTERISTICS")
        report.append("-"*80)
        report.append("Sequence Type: Genomic DNA sequences (reference and alternate)")
        report.append("Sequence Length: 1024 base pairs (configured)")
        report.append("Encoding: Pre-tokenized using DNABERT-2 tokenizer")
        report.append("PCA Components: 16 (for dimensionality reduction)")
        report.append("")

        report.append("-"*80)
        report.append("5. BIOLOGICAL CONTEXT")
        report.append("-"*80)
        report.append("sQTL (Splicing Quantitative Trait Loci):")
        report.append("  - Genetic variants that affect RNA splicing patterns")
        report.append("  - GTEx v8: Genotype-Tissue Expression project version 8")
        report.append("  - Whole_Blood: Specific tissue type analyzed")
        report.append("  - Significance: Based on p-value threshold comparison")
        report.append("")
        report.append("Classification Task:")
        report.append("  - Significant: pval_nominal < pval_nominal_threshold")
        report.append("  - Not Significant: pval_nominal >= pval_nominal_threshold")
        report.append("")

        report.append("-"*80)
        report.append("6. MODEL TRAINING DETAILS")
        report.append("-"*80)
        report.append("Architecture: DNABERT-2-117M + Classification Head")
        report.append("Training Configuration:")
        report.append("  - Epochs: 100")
        report.append("  - GPUs: 1")
        report.append("  - Workers: 8")
        report.append("  - Task: classification")
        report.append("  - Output Size: 2 (binary classification)")
        report.append("  - Target: p_val")
        report.append("")

        report.append("-"*80)
        report.append("7. KEY FINDINGS")
        report.append("-"*80)

        # Class balance finding
        if imbalance_ratio > 3:
            report.append("⚠️  Class Imbalance:")
            report.append(f"   Significant class imbalance detected ({imbalance_ratio:.1f}:1 ratio)")
            report.append("   Recommendation: Consider using class weights or resampling")
        else:
            report.append("✓  Class Balance: Relatively balanced dataset")
        report.append("")

        report.append("-"*80)
        report.append("8. RECOMMENDATIONS")
        report.append("-"*80)
        report.append("Data Handling:")
        report.append("  - Data is pre-tokenized and stored in numpy format")
        report.append("  - Use memory mapping for efficient loading")
        report.append("  - Consider data augmentation for imbalanced classes")
        report.append("")
        report.append("Model Evaluation:")
        report.append("  - Use stratified splits to maintain class distribution")
        report.append("  - Monitor both accuracy and F1-score due to class imbalance")
        report.append("  - Consider precision-recall curves for model assessment")
        report.append("")

        report.append("="*80)
        report.append("End of Report")
        report.append("="*80)

        report_text = "\n".join(report)
        print(report_text)

        with open(self.output_dir / "comprehensive_summary.txt", "w") as f:
            f.write(report_text)

        print(f"\n✓ Saved: {self.output_dir / 'comprehensive_summary.txt'}")

    def run_full_eda(self):
        """Run complete EDA pipeline"""
        print("\n" + "="*80)
        print("STARTING COMPREHENSIVE EDA FOR sQTL TRAINING DATA")
        print("="*80)

        # Load data
        self.load_data()

        if len(self.sequences) == 0:
            print("\n⚠️  ERROR: No data loaded. Cannot proceed with EDA.")
            return

        # Run analyses
        self.analyze_dataset_overview()
        self.analyze_class_distribution()
        self.analyze_sequence_characteristics()
        self.analyze_data_quality()
        self.generate_summary_report()

        print("\n" + "="*80)
        print("EDA COMPLETE!")
        print("="*80)
        print(f"\nAll outputs saved to: {self.output_dir.absolute()}")
        print("\nGenerated files:")
        for file in sorted(self.output_dir.glob("*")):
            print(f"  - {file.name}")


def main():
    """Main execution function"""
    # Path to training data
    data_path = "/project/def-mahadeva/ranaab/genomic-FM/root/data/npy_output_delta_sqtl_pval_dnabert2"

    # Create explorer and run EDA
    explorer = sQTLDataExplorer(data_path)
    explorer.run_full_eda()


if __name__ == "__main__":
    main()
