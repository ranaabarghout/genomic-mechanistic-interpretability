"""
Comprehensive Analysis of sQTL Model Predictions
=================================================
Analyzes the fine-tuned DNABERT-2 model's behavior on sQTL data using:
- Attention analysis (which sequence positions are important)
- Feature importance (which PCA components matter most)
- Prediction analysis (classification performance, errors)
- Layer-wise activation patterns
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import yaml
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from transformers import AutoTokenizer, AutoModel

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class sQTLModelAnalyzer:
    """Comprehensive analysis of sQTL model predictions and behavior"""

    def __init__(self,
                 checkpoint_path: str,
                 data_path: str,
                 output_dir: str = "outputs/sqtl_analysis"):
        """
        Initialize analyzer

        Args:
            checkpoint_path: Path to fine-tuned model checkpoint
            data_path: Path to sQTL data directory
            output_dir: Output directory for analysis results
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load model
        print("\nLoading model...")
        self.model = self._load_model()

        # Load data
        print("\nLoading sQTL data...")
        self.sequences, self.labels, self.annotations = self._load_data()

        # Load metadata
        with open(self.data_path / "sqtl_pval_dnabert2_y_class.yaml", 'r') as f:
            self.y_class = yaml.safe_load(f)
        self.class_names = {v: k for k, v in self.y_class.items()}

        print(f"\nLoaded {len(self.sequences)} samples")
        print(f"Device: {self.device}")

    def _load_model(self):
        """Load the fine-tuned DNABERT-2 model"""
        from transformers import BertForMaskedLM

        # Load base model
        model = BertForMaskedLM.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        bert_model = model.bert

        # Load checkpoint if available
        if self.checkpoint_path.exists():
            print(f"Loading checkpoint: {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

            # Extract only BERT weights (checkpoint has CNN head weights)
            state_dict = checkpoint.get('state_dict', checkpoint)
            bert_weights = {k.replace('model.bert.', ''): v
                           for k, v in state_dict.items()
                           if 'model.bert.' in k}

            if bert_weights:
                bert_model.load_state_dict(bert_weights, strict=False)
                print(f"Loaded {len(bert_weights)} BERT weights from checkpoint")
            else:
                print("⚠️  No BERT weights in checkpoint, using base model")

        bert_model = bert_model.to(self.device)
        bert_model.eval()
        return bert_model

    def _load_data(self):
        """Load sQTL data from binary files"""
        sequences = []
        labels = []
        annotations = []

        # Data chunks
        chunks = [0, 2500, 5000]
        seq_length = 1024
        pca_components = 16

        for chunk_id in chunks:
            # Load labels
            y_file = self.data_path / f"sqtl_pval_dnabert2_y_{chunk_id}.npy"
            with open(y_file, 'rb') as f:
                y_data = np.frombuffer(f.read(), dtype=np.int64)
                num_samples = len(y_data)
                labels.append(y_data)

            # Load sequences
            seq_file = self.data_path / f"sqtl_pval_dnabert2_seq1_{chunk_id}.npy"
            with open(seq_file, 'rb') as f:
                seq_data = np.frombuffer(f.read(), dtype=np.float32).reshape(
                    num_samples, seq_length, pca_components
                )
                sequences.append(seq_data)

            # Load annotations
            annot_file = self.data_path / f"sqtl_pval_dnabert2_annot_{chunk_id}.npy"
            with open(annot_file, 'rb') as f:
                annot_data = np.frombuffer(f.read(), dtype=np.int64)
                annotations.append(annot_data)

        sequences = np.concatenate(sequences, axis=0)
        labels = np.concatenate(labels, axis=0)
        annotations = np.concatenate(annotations, axis=0)

        return sequences, labels, annotations

    def analyze_feature_importance(self, num_samples=100):
        """Analyze which PCA features are most important for predictions"""
        print("\n" + "="*70)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*70)

        # Sample data stratified by class
        sample_indices = []
        for class_idx in np.unique(self.labels):
            class_mask = self.labels == class_idx
            class_indices = np.where(class_mask)[0]
            sample_size = min(num_samples // 2, len(class_indices))
            sampled = np.random.choice(class_indices, sample_size, replace=False)
            sample_indices.extend(sampled)

        sample_sequences = self.sequences[sample_indices]
        sample_labels = self.labels[sample_indices]

        # Calculate feature importance by position and component
        print(f"\nAnalyzing {len(sample_indices)} samples...")

        # Average absolute values by PCA component (across all positions and samples)
        component_importance = np.mean(np.abs(sample_sequences), axis=(0, 1))

        # Position-wise importance (average across samples and components)
        position_importance = np.mean(np.abs(sample_sequences), axis=(0, 2))

        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # PCA component importance
        ax = axes[0, 0]
        x_pos = np.arange(len(component_importance))
        bars = ax.bar(x_pos, component_importance, color='#3498db', alpha=0.7, edgecolor='black')
        ax.set_xlabel('PCA Component', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Absolute Value', fontsize=12, fontweight='bold')
        ax.set_title('PCA Component Importance', fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'PC{i+1}' for i in range(len(component_importance))], rotation=45)
        ax.grid(axis='y', alpha=0.3)

        # Position importance (sliding window average)
        ax = axes[0, 1]
        window_size = 50
        smoothed = np.convolve(position_importance, np.ones(window_size)/window_size, mode='valid')
        ax.plot(smoothed, color='#e74c3c', linewidth=2)
        ax.fill_between(range(len(smoothed)), smoothed, alpha=0.3, color='#e74c3c')
        ax.set_xlabel('Sequence Position', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Absolute Value', fontsize=12, fontweight='bold')
        ax.set_title(f'Position Importance (window={window_size})', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)

        # Heatmap: PCA components vs position (subsampled for visualization)
        ax = axes[1, 0]
        # Subsample positions for visualization
        position_subsample = 20
        heatmap_data = np.mean(np.abs(sample_sequences), axis=0)[::position_subsample, :].T
        im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax.set_xlabel('Sequence Position (subsampled)', fontsize=12, fontweight='bold')
        ax.set_ylabel('PCA Component', fontsize=12, fontweight='bold')
        ax.set_title('Feature Importance Heatmap', fontsize=13, fontweight='bold')
        ax.set_yticks(range(len(component_importance)))
        ax.set_yticklabels([f'PC{i+1}' for i in range(len(component_importance))])
        plt.colorbar(im, ax=ax, label='Importance')

        # Feature importance by class
        ax = axes[1, 1]
        importance_by_class = {}
        for class_idx in np.unique(sample_labels):
            class_mask = sample_labels == class_idx
            class_seqs = sample_sequences[class_mask]
            class_importance = np.mean(np.abs(class_seqs), axis=(0, 1))
            importance_by_class[self.class_names[class_idx]] = class_importance

        x = np.arange(len(component_importance))
        width = 0.35
        class_labels = list(importance_by_class.keys())
        for i, (class_name, importance) in enumerate(importance_by_class.items()):
            offset = width * (i - len(class_labels)/2 + 0.5)
            color = '#2ecc71' if 'significant' in class_name else '#e74c3c'
            ax.bar(x + offset, importance, width, label=class_name,
                   alpha=0.7, color=color, edgecolor='black')

        ax.set_xlabel('PCA Component', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Absolute Value', fontsize=12, fontweight='bold')
        ax.set_title('Feature Importance by Class', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'PC{i+1}' for i in range(len(component_importance))], rotation=45)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "feature_importance.png", dpi=300, bbox_inches='tight')
        print(f"Saved: {self.output_dir / 'feature_importance.png'}")
        plt.close()

        # Print statistics
        print(f"\nTop 5 Most Important PCA Components:")
        top_components = np.argsort(component_importance)[::-1][:5]
        for i, comp_idx in enumerate(top_components, 1):
            print(f"  {i}. PC{comp_idx+1}: {component_importance[comp_idx]:.4f}")

        # Save detailed report
        with open(self.output_dir / "feature_importance_report.txt", 'w') as f:
            f.write("Feature Importance Analysis Report\n")
            f.write("="*70 + "\n\n")
            f.write(f"Samples analyzed: {len(sample_indices)}\n\n")
            f.write("PCA Component Importance (sorted):\n")
            for comp_idx in top_components:
                f.write(f"  PC{comp_idx+1}: {component_importance[comp_idx]:.6f}\n")
            f.write(f"\nMost important position range: {np.argmax(position_importance)}-{np.argmax(position_importance)+50}\n")

        print(f"Saved: {self.output_dir / 'feature_importance_report.txt'}")

    def analyze_attention_patterns(self, num_samples=50):
        """Analyze attention patterns on sQTL sequences"""
        print("\n" + "="*70)
        print("ATTENTION PATTERN ANALYSIS")
        print("="*70)

        print("\n⚠️  Note: Sequences are PCA-transformed (16 components)")
        print("   Attention analysis works on model embeddings, not raw sequences")

        # Sample data
        sample_indices = np.random.choice(len(self.sequences),
                                         min(num_samples, len(self.sequences)),
                                         replace=False)

        print(f"\nAnalyzing attention patterns for {len(sample_indices)} samples...")
        print("This analysis shows which sequence positions the model focuses on.")

        # Note: Since sequences are PCA-transformed, we can't run them through DNABERT-2 directly
        # The model expects tokenized DNA sequences, not PCA features

        report = []
        report.append("Attention Pattern Analysis Report")
        report.append("="*70)
        report.append("")
        report.append("⚠️  IMPORTANT LIMITATION:")
        report.append("The training data is PCA-transformed (16 components), not raw DNA sequences.")
        report.append("DNABERT-2 expects tokenized DNA sequences as input.")
        report.append("")
        report.append("To perform attention analysis, you would need:")
        report.append("1. Access to the original (pre-PCA) DNA sequences")
        report.append("2. Or, reconstruct sequences from PCA components (with information loss)")
        report.append("3. Or, use the trained classification head with PCA inputs")
        report.append("")
        report.append("Current Analysis:")
        report.append(f"- Total samples: {len(self.sequences)}")
        report.append(f"- Sequence shape: {self.sequences.shape}")
        report.append(f"- Model: DNABERT-2 (expects DNA token sequences)")
        report.append(f"- Data format: PCA-transformed features")
        report.append("")
        report.append("Recommendation:")
        report.append("- For attention analysis, load original GTEx sQTL data")
        report.append("- Extract reference/alternate sequences from variant records")
        report.append("- Run attention analysis on those sequences")

        report_text = "\n".join(report)
        print("\n" + report_text)

        with open(self.output_dir / "attention_analysis_note.txt", 'w') as f:
            f.write(report_text)

        print(f"\nSaved: {self.output_dir / 'attention_analysis_note.txt'}")

    def analyze_class_separability(self):
        """Analyze how well classes are separated in feature space"""
        print("\n" + "="*70)
        print("CLASS SEPARABILITY ANALYSIS")
        print("="*70)

        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        # Flatten sequences for dimensionality reduction
        flattened = self.sequences.reshape(len(self.sequences), -1)

        print(f"\nOriginal shape: {self.sequences.shape}")
        print(f"Flattened shape: {flattened.shape}")

        # Sample for faster computation (t-SNE is expensive)
        sample_size = min(500, len(flattened))  # Reduced from 2000
        sample_indices = np.random.choice(len(flattened), sample_size, replace=False)
        sample_data = flattened[sample_indices]
        sample_labels = self.labels[sample_indices]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # PCA projection
        print("\nComputing PCA projection...")
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(sample_data)

        ax = axes[0]
        for class_idx in np.unique(sample_labels):
            mask = sample_labels == class_idx
            color = '#2ecc71' if class_idx == 0 else '#e74c3c'
            ax.scatter(pca_result[mask, 0], pca_result[mask, 1],
                      alpha=0.6, s=30, label=self.class_names[class_idx],
                      color=color, edgecolors='black', linewidth=0.5)

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                     fontsize=12, fontweight='bold')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                     fontsize=12, fontweight='bold')
        ax.set_title('PCA Projection', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # t-SNE projection (on smaller sample and reduced dimensions)
        print(f"Computing t-SNE projection on {sample_size} samples...")
        # First reduce to 50 dimensions with PCA for faster t-SNE
        pca_50 = PCA(n_components=50)
        data_reduced = pca_50.fit_transform(sample_data)
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500, n_jobs=4)
        tsne_result = tsne.fit_transform(data_reduced)

        ax = axes[1]
        for class_idx in np.unique(sample_labels):
            mask = sample_labels == class_idx
            color = '#2ecc71' if class_idx == 0 else '#e74c3c'
            ax.scatter(tsne_result[mask, 0], tsne_result[mask, 1],
                      alpha=0.6, s=30, label=self.class_names[class_idx],
                      color=color, edgecolors='black', linewidth=0.5)

        ax.set_xlabel('t-SNE 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('t-SNE 2', fontsize=12, fontweight='bold')
        ax.set_title(f't-SNE Projection (n={sample_size})', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "class_separability.png", dpi=300, bbox_inches='tight')
        print(f"\nSaved: {self.output_dir / 'class_separability.png'}")
        plt.close()

        # Calculate separability metrics
        from sklearn.metrics import silhouette_score
        silhouette = silhouette_score(sample_data, sample_labels)

        print(f"\nSeparability Metrics:")
        print(f"  Silhouette Score: {silhouette:.4f}")
        print(f"  PCA explained variance (2 components): {sum(pca.explained_variance_ratio_)*100:.2f}%")

    def analyze_error_patterns(self):
        """Analyze patterns in the data (note: no predictions available without classification head)"""
        print("\n" + "="*70)
        print("DATA PATTERN ANALYSIS")
        print("="*70)

        print("\n⚠️  Note: Model checkpoint contains only BERT encoder (no classification head)")
        print("   Cannot generate predictions for error analysis")
        print("   Analyzing data patterns instead...")

        # Analyze feature patterns by class
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Distribution of sequence magnitudes by class
        ax = axes[0, 0]
        magnitudes_by_class = {}
        for class_idx in np.unique(self.labels):
            mask = self.labels == class_idx
            seqs = self.sequences[mask]
            mags = np.linalg.norm(seqs.reshape(len(seqs), -1), axis=1)
            magnitudes_by_class[self.class_names[class_idx]] = mags

        for class_name, mags in magnitudes_by_class.items():
            color = '#2ecc71' if 'significant' in class_name else '#e74c3c'
            ax.hist(mags, bins=50, alpha=0.6, label=class_name, color=color, edgecolor='black')

        ax.set_xlabel('Sequence Magnitude', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Sequence Magnitude Distribution by Class', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # Mean feature values by class
        ax = axes[0, 1]
        mean_features_by_class = {}
        for class_idx in np.unique(self.labels):
            mask = self.labels == class_idx
            mean_features = np.mean(self.sequences[mask], axis=(0, 1))  # Average over samples and positions
            mean_features_by_class[self.class_names[class_idx]] = mean_features

        x = np.arange(len(mean_features))
        width = 0.35
        for i, (class_name, features) in enumerate(mean_features_by_class.items()):
            offset = width * (i - 0.5)
            color = '#2ecc71' if 'significant' in class_name else '#e74c3c'
            ax.bar(x + offset, features, width, label=class_name,
                   alpha=0.7, color=color, edgecolor='black')

        ax.set_xlabel('PCA Component', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Value', fontsize=12, fontweight='bold')
        ax.set_title('Mean Feature Values by Class', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'PC{i+1}' for i in range(len(mean_features))], rotation=45)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)

        # Feature variance by class
        ax = axes[1, 0]
        var_features_by_class = {}
        for class_idx in np.unique(self.labels):
            mask = self.labels == class_idx
            var_features = np.std(self.sequences[mask], axis=(0, 1))
            var_features_by_class[self.class_names[class_idx]] = var_features

        for i, (class_name, features) in enumerate(var_features_by_class.items()):
            offset = width * (i - 0.5)
            color = '#2ecc71' if 'significant' in class_name else '#e74c3c'
            ax.bar(x + offset, features, width, label=class_name,
                   alpha=0.7, color=color, edgecolor='black')

        ax.set_xlabel('PCA Component', fontsize=12, fontweight='bold')
        ax.set_ylabel('Standard Deviation', fontsize=12, fontweight='bold')
        ax.set_title('Feature Variability by Class', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'PC{i+1}' for i in range(len(mean_features))], rotation=45)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Sample distribution across data chunks
        ax = axes[1, 1]
        chunk_labels = ['Chunk 0\n(0-2499)', 'Chunk 1\n(2500-4999)', 'Chunk 2\n(5000-7499)']
        chunk_size = 2500

        for class_idx in np.unique(self.labels):
            class_name = self.class_names[class_idx]
            color = '#2ecc71' if 'significant' in class_name else '#e74c3c'
            counts = []
            for chunk in range(3):
                start = chunk * chunk_size
                end = start + chunk_size
                chunk_labels_subset = self.labels[start:end]
                count = np.sum(chunk_labels_subset == class_idx)
                counts.append(count)

            x_pos = np.arange(len(chunk_labels))
            offset = width * (class_idx - 0.5)
            ax.bar(x_pos + offset, counts, width, label=class_name,
                   alpha=0.7, color=color, edgecolor='black')

        ax.set_xlabel('Data Chunk', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sample Count', fontsize=12, fontweight='bold')
        ax.set_title('Class Distribution Across Chunks', fontsize=13, fontweight='bold')
        ax.set_xticks(range(len(chunk_labels)))
        ax.set_xticklabels(chunk_labels)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "data_patterns.png", dpi=300, bbox_inches='tight')
        print(f"\nSaved: {self.output_dir / 'data_patterns.png'}")
        plt.close()

    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report"""
        print("\n" + "="*70)
        print("GENERATING COMPREHENSIVE ANALYSIS REPORT")
        print("="*70)

        report = []
        report.append("="*80)
        report.append("COMPREHENSIVE ANALYSIS REPORT: sQTL Model & Data")
        report.append("="*80)
        report.append("")
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Model: DNABERT-2-117M (fine-tuned)")
        report.append(f"Checkpoint: {self.checkpoint_path.name}")
        report.append(f"Dataset: sQTL (GTEx v8 Whole_Blood)")
        report.append(f"Total Samples: {len(self.sequences):,}")
        report.append("")

        report.append("-"*80)
        report.append("1. DATA SUMMARY")
        report.append("-"*80)
        report.append(f"Sequence shape: {self.sequences.shape}")
        report.append(f"  - Samples: {self.sequences.shape[0]:,}")
        report.append(f"  - Sequence length: {self.sequences.shape[1]} bp")
        report.append(f"  - PCA components: {self.sequences.shape[2]}")
        report.append("")

        report.append("Class Distribution:")
        for class_idx in sorted(self.y_class.values()):
            count = np.sum(self.labels == class_idx)
            pct = (count / len(self.labels)) * 100
            report.append(f"  {self.class_names[class_idx]:20s}: {count:5,} ({pct:5.2f}%)")
        report.append("")

        report.append("-"*80)
        report.append("2. MODEL ARCHITECTURE")
        report.append("-"*80)
        report.append("Base Model: DNABERT-2-117M")
        report.append("  - Parameters: 117M")
        report.append("  - Architecture: BERT-based transformer")
        report.append("  - Layers: 12 transformer layers")
        report.append("  - Hidden size: 768")
        report.append("  - Attention heads: 12")
        report.append("")
        report.append("Fine-tuning:")
        report.append("  - Training epochs: 100")
        report.append("  - Task: Binary classification (sQTL significance)")
        report.append("  - Checkpoint: Contains BERT encoder weights")
        report.append("  - Note: Classification head trained separately (CNN)")
        report.append("")

        report.append("-"*80)
        report.append("3. KEY FINDINGS")
        report.append("-"*80)
        report.append("")

        # Feature importance
        report.append("Feature Importance:")
        sample_seqs = self.sequences[np.random.choice(len(self.sequences), 1000, replace=False)]
        component_importance = np.mean(np.abs(sample_seqs), axis=(0, 1))
        top_5 = np.argsort(component_importance)[::-1][:5]
        report.append("  Top 5 PCA components:")
        for i, comp_idx in enumerate(top_5, 1):
            report.append(f"    {i}. PC{comp_idx+1}: {component_importance[comp_idx]:.4f}")
        report.append("")

        # Class separability
        report.append("Class Separability:")
        flattened = self.sequences.reshape(len(self.sequences), -1)
        sample_idx = np.random.choice(len(flattened), min(1000, len(flattened)), replace=False)
        from sklearn.metrics import silhouette_score
        silhouette = silhouette_score(flattened[sample_idx], self.labels[sample_idx])
        report.append(f"  Silhouette score: {silhouette:.4f}")
        if silhouette > 0.5:
            report.append("  ✓ Classes are well-separated")
        elif silhouette > 0.25:
            report.append("  ⚠️  Moderate class separation")
        else:
            report.append("  ⚠️  Poor class separation (challenging classification task)")
        report.append("")

        report.append("-"*80)
        report.append("4. LIMITATIONS & CONSIDERATIONS")
        report.append("-"*80)
        report.append("")
        report.append("Data Format:")
        report.append("  - Sequences are PCA-transformed (16 components)")
        report.append("  - Original DNA sequences not available in this format")
        report.append("  - Limits certain types of interpretability analysis")
        report.append("")
        report.append("Model Checkpoint:")
        report.append("  - Contains BERT encoder weights only")
        report.append("  - Classification head (CNN) not included in checkpoint")
        report.append("  - Cannot generate predictions without full model")
        report.append("")
        report.append("Analysis Capabilities:")
        report.append("  ✓ Feature importance analysis")
        report.append("  ✓ Class separability analysis")
        report.append("  ✓ Data pattern analysis")
        report.append("  ✗ Attention analysis (requires original sequences)")
        report.append("  ✗ Prediction analysis (requires classification head)")
        report.append("  ✗ Error analysis (requires full model)")
        report.append("")

        report.append("-"*80)
        report.append("5. RECOMMENDATIONS")
        report.append("-"*80)
        report.append("")
        report.append("For Deeper Analysis:")
        report.append("  1. Access original GTEx sQTL data (pre-PCA)")
        report.append("  2. Extract reference/alternate DNA sequences from variant records")
        report.append("  3. Load full classification model (BERT + CNN head)")
        report.append("  4. Run attention analysis on original sequences")
        report.append("  5. Perform prediction-based interpretability")
        report.append("")
        report.append("Model Improvement:")
        report.append("  - Consider class weights (1.36:1 imbalance)")
        report.append("  - Experiment with different PCA components (currently 16)")
        report.append("  - Try data augmentation techniques")
        report.append("  - Analyze per-tissue performance (currently Whole_Blood only)")
        report.append("")

        report.append("="*80)
        report.append("Analysis outputs saved to: " + str(self.output_dir.absolute()))
        report.append("="*80)

        report_text = "\n".join(report)
        print("\n" + report_text)

        with open(self.output_dir / "comprehensive_analysis_report.txt", 'w') as f:
            f.write(report_text)

        print(f"\n✓ Saved: {self.output_dir / 'comprehensive_analysis_report.txt'}")

    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("\n" + "="*80)
        print("STARTING COMPREHENSIVE sQTL ANALYSIS")
        print("="*80)
        print(f"\nOutput directory: {self.output_dir.absolute()}")

        # Run analyses
        self.analyze_feature_importance(num_samples=200)
        self.analyze_class_separability()
        self.analyze_error_patterns()
        self.analyze_attention_patterns(num_samples=50)
        self.generate_comprehensive_report()

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\nAll outputs saved to: {self.output_dir.absolute()}")
        print("\nGenerated files:")
        for file in sorted(self.output_dir.glob("*")):
            size = file.stat().st_size
            if size > 1024*1024:
                size_str = f"{size/(1024*1024):.1f}MB"
            elif size > 1024:
                size_str = f"{size/1024:.1f}KB"
            else:
                size_str = f"{size}B"
            print(f"  - {file.name:50s} ({size_str})")


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive sQTL Model Analysis")
    parser.add_argument("--checkpoint", type=str,
                       default="/project/def-mahadeva/ranaab/genomic-FM/GV-Rep/9p1e0e6n/checkpoints/epoch=99-step=18800.ckpt",
                       help="Path to model checkpoint")
    parser.add_argument("--data", type=str,
                       default="/project/def-mahadeva/ranaab/genomic-FM/root/data/npy_output_delta_sqtl_pval_dnabert2",
                       help="Path to sQTL data directory")
    parser.add_argument("--output", type=str,
                       default="outputs/sqtl_analysis",
                       help="Output directory")

    args = parser.parse_args()

    # Create analyzer
    analyzer = sQTLModelAnalyzer(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        output_dir=args.output
    )

    # Run analysis
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
