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
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


class HiddenStateAnalyzer:
    """Analyze hidden state representations for mechanistic interpretability"""

    def __init__(self,
                 model,
                 tokenizer,
                 config,
                 data_loader,
                 samples: List,
                 output_dir: str,
                 device: str = None):
        """
        Initialize analyzer

        Args:
            model: Pre-loaded model
            tokenizer: Pre-loaded tokenizer
            config: Model config
            data_loader: Data loader with label mapping
            samples: List of samples
            output_dir: Output directory
            device: Device to use (auto-detect if None)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.data_loader = data_loader
        self.samples = samples
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Separate by class
        self.class_0_samples = [s for s in self.samples if s.label == 0]
        self.class_1_samples = [s for s in self.samples if s.label == 1]

    def get_hidden_states(self, sequence: str, max_length: int = 512):
        """
        Get hidden states from model

        Returns:
            Dictionary with:
            - last_hidden_state: (seq_len, hidden_size)
            - pooler_output: (hidden_size,)
            - tokens: List of token strings
        """
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding="max_length"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # DNABERT-2 returns (last_hidden_state, pooler_output)
        last_hidden_state = outputs[0][0].cpu().numpy()  # (seq_len, hidden_size)
        pooler_output = outputs[1][0].cpu().numpy()  # (hidden_size,)

        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        return {
            'last_hidden_state': last_hidden_state,
            'pooler_output': pooler_output,
            'tokens': tokens,
            'attention_mask': inputs['attention_mask'][0].cpu().numpy()
        }

    def analyze_cls_representation(self, num_samples: int = None):
        """
        Analyze CLS token representations across classes

        CLS token typically contains sequence-level information
        """
        print("\n" + "="*70)
        print("CLS TOKEN REPRESENTATION ANALYSIS")
        print("="*70)

        if num_samples is None:
            num_samples = min(len(self.class_0_samples), len(self.class_1_samples))

        class_0_cls = []
        class_1_cls = []

        print(f"\nExtracting CLS representations from {num_samples} samples per class...")

        for samples, storage in [
            (self.class_0_samples[:num_samples], class_0_cls),
            (self.class_1_samples[:num_samples], class_1_cls)
        ]:
            for sample in tqdm(samples):
                result = self.get_hidden_states(sample.ref_sequence)
                # CLS token is at position 0
                cls_hidden = result['last_hidden_state'][0]
                storage.append(cls_hidden)

        class_0_cls = np.array(class_0_cls)  # (n_samples, hidden_size)
        class_1_cls = np.array(class_1_cls)

        # PCA visualization
        print("\nPerforming PCA dimensionality reduction...")
        all_cls = np.vstack([class_0_cls, class_1_cls])
        pca = PCA(n_components=2)
        cls_pca = pca.fit_transform(all_cls)

        n0 = len(class_0_cls)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # PCA plot
        axes[0].scatter(cls_pca[:n0, 0], cls_pca[:n0, 1],
                       alpha=0.6, label=self.data_loader.get_label_name(0), s=50)
        axes[0].scatter(cls_pca[n0:, 0], cls_pca[n0:, 1],
                       alpha=0.6, label=self.data_loader.get_label_name(1), s=50)
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
        axes[0].set_title('CLS Token Representations (PCA)', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Distance analysis
        class_0_mean = class_0_cls.mean(axis=0)
        class_1_mean = class_1_cls.mean(axis=0)

        # Compute distances to class centroids
        dist_to_c0 = [np.linalg.norm(x - class_0_mean) for x in class_0_cls]
        dist_to_c1 = [np.linalg.norm(x - class_1_mean) for x in class_1_cls]

        axes[1].hist(dist_to_c0, bins=20, alpha=0.6,
                    label=f'{self.data_loader.get_label_name(0)} to centroid')
        axes[1].hist(dist_to_c1, bins=20, alpha=0.6,
                    label=f'{self.data_loader.get_label_name(1)} to centroid')
        axes[1].set_xlabel('Distance to class centroid')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Within-class Compactness', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / '1_cls_representations.png', dpi=150, bbox_inches='tight')
        print(f"Saved: 1_cls_representations.png")

        # Statistical comparison
        between_class_dist = np.linalg.norm(class_0_mean - class_1_mean)
        within_class_0 = np.mean(dist_to_c0)
        within_class_1 = np.mean(dist_to_c1)

        print(f"\nRepresentation statistics:")
        print(f"  Between-class distance: {between_class_dist:.3f}")
        print(f"  Within-class 0 distance: {within_class_0:.3f}")
        print(f"  Within-class 1 distance: {within_class_1:.3f}")
        print(f"  Separation ratio: {between_class_dist / ((within_class_0 + within_class_1) / 2):.3f}")

        return {
            'class_0_cls': class_0_cls,
            'class_1_cls': class_1_cls,
            'pca': pca,
            'between_class_dist': between_class_dist,
            'within_class_dists': (within_class_0, within_class_1)
        }

    def analyze_variant_position_states(self, num_samples: int = None):
        """
        Analyze hidden states at variant positions
        """
        print("\n" + "="*70)
        print("VARIANT POSITION HIDDEN STATE ANALYSIS")
        print("="*70)

        if num_samples is None:
            num_samples = min(len(self.class_0_samples), len(self.class_1_samples))

        class_0_variant = []
        class_1_variant = []

        print(f"\nExtracting variant position states from {num_samples} samples per class...")

        for samples, storage in [
            (self.class_0_samples[:num_samples], class_0_variant),
            (self.class_1_samples[:num_samples], class_1_variant)
        ]:
            for sample in tqdm(samples):
                if sample.variant_pos < 0:
                    continue

                result = self.get_hidden_states(sample.ref_sequence)

                # Get variant token position (approximate with k-mer size)
                kmer_size = 6
                variant_token = min(1 + (sample.variant_pos // kmer_size),
                                  len(result['tokens']) - 1)

                variant_hidden = result['last_hidden_state'][variant_token]
                storage.append(variant_hidden)

        class_0_variant = np.array(class_0_variant)
        class_1_variant = np.array(class_1_variant)

        print(f"\nValid samples: Class 0={len(class_0_variant)}, Class 1={len(class_1_variant)}")

        # Per-dimension statistical test
        p_values = []
        effect_sizes = []

        for dim in range(self.config.hidden_size):
            t_stat, p_val = stats.ttest_ind(class_0_variant[:, dim], class_1_variant[:, dim])
            p_values.append(p_val)

            pooled_std = np.sqrt((np.var(class_0_variant[:, dim]) +
                                 np.var(class_1_variant[:, dim])) / 2)
            effect_size = (np.mean(class_0_variant[:, dim]) -
                          np.mean(class_1_variant[:, dim])) / (pooled_std + 1e-10)
            effect_sizes.append(effect_size)

        p_values = np.array(p_values)
        effect_sizes = np.array(effect_sizes)

        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # P-value distribution
        axes[0].hist(-np.log10(p_values + 1e-10), bins=50, edgecolor='black')
        axes[0].axvline(-np.log10(0.05), color='red', linestyle='--',
                       label='p=0.05 threshold')
        axes[0].set_xlabel('-log10(p-value)')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Statistical Significance per Dimension', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Effect size distribution
        axes[1].hist(effect_sizes, bins=50, edgecolor='black')
        axes[1].axvline(0, color='red', linestyle='--')
        axes[1].set_xlabel("Cohen's d")
        axes[1].set_ylabel('Count')
        axes[1].set_title('Effect Sizes per Dimension', fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        # Top discriminative dimensions - plot pairwise scatter
        top_dims = np.argsort(p_values)[:20]

        # Get minimum size for fair comparison
        min_size = min(len(class_0_variant), len(class_1_variant))

        for dim in top_dims[:5]:
            # Only plot up to minimum size to avoid size mismatch
            axes[2].scatter(class_0_variant[:min_size, dim], class_1_variant[:min_size, dim],
                          alpha=0.5, s=30, label=f'Dim {dim}')

        # Plot diagonal line
        all_vals = np.concatenate([class_0_variant[:min_size, top_dims[0]],
                                   class_1_variant[:min_size, top_dims[0]]])
        axes[2].plot([all_vals.min(), all_vals.max()],
                    [all_vals.min(), all_vals.max()],
                    'r--', alpha=0.5)
        axes[2].set_xlabel(f'{self.data_loader.get_label_name(0)} values')
        axes[2].set_ylabel(f'{self.data_loader.get_label_name(1)} values')
        axes[2].set_title('Top 5 Discriminative Dimensions', fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / '2_variant_position_states.png', dpi=150, bbox_inches='tight')
        print(f"Saved: 2_variant_position_states.png")

        sig_dims = (p_values < 0.01).sum()
        print(f"\nSignificant dimensions (p < 0.01): {sig_dims}/{self.config.hidden_size}")
        print(f"Mean absolute effect size: {np.abs(effect_sizes).mean():.4f}")

        return {
            'class_0_variant': class_0_variant,
            'class_1_variant': class_1_variant,
            'p_values': p_values,
            'effect_sizes': effect_sizes,
            'top_discriminative_dims': top_dims[:20]
        }

    def analyze_sequence_position_evolution(self, num_samples: int = 5):
        """
        Visualize how hidden states evolve across sequence positions
        """
        print("\n" + "="*70)
        print("SEQUENCE POSITION EVOLUTION")
        print("="*70)

        fig, axes = plt.subplots(2, num_samples, figsize=(5*num_samples, 8))
        if num_samples == 1:
            axes = axes.reshape(-1, 1)

        for class_idx, (samples, class_name) in enumerate([
            (self.class_0_samples[:num_samples], self.data_loader.get_label_name(0)),
            (self.class_1_samples[:num_samples], self.data_loader.get_label_name(1))
        ]):
            for sample_idx in range(min(num_samples, len(samples))):
                sample = samples[sample_idx]
                result = self.get_hidden_states(sample.ref_sequence)

                hidden = result['last_hidden_state']  # (seq_len, hidden_size)

                ax = axes[class_idx, sample_idx]

                # Plot heatmap of hidden states
                im = ax.imshow(hidden.T, aspect='auto', cmap='RdBu_r',
                             vmin=-2, vmax=2, interpolation='nearest')

                # Mark variant position
                if sample.variant_pos >= 0:
                    kmer_size = 6
                    variant_token = min(1 + (sample.variant_pos // kmer_size),
                                      hidden.shape[0] - 1)
                    ax.axvline(variant_token, color='yellow', linestyle='--', linewidth=2)

                ax.set_title(f'{class_name}\nSample {sample_idx+1}', fontweight='bold')
                ax.set_xlabel('Sequence Position (tokens)')
                ax.set_ylabel('Hidden Dimension')

                if sample_idx == num_samples - 1:
                    plt.colorbar(im, ax=ax, fraction=0.046)

        plt.tight_layout()
        plt.savefig(self.output_dir / '3_sequence_evolution.png', dpi=150, bbox_inches='tight')
        print(f"Saved: 3_sequence_evolution.png")

    def run_full_analysis(self):
        """Run complete hidden state analysis"""
        print("\n" + "="*70)
        print("HIDDEN STATE MECHANISTIC ANALYSIS")
        print("="*70)
        print(f"Output: {self.output_dir}")
        print("="*70)

        results = {}

        # 1. CLS representation analysis
        print("\n" + "-"*70)
        print("Step 1/3: Analyzing CLS token representations")
        print("-"*70)
        results['cls_analysis'] = self.analyze_cls_representation(num_samples=50)

        # 2. Variant position analysis
        print("\n" + "-"*70)
        print("Step 2/3: Analyzing variant position states")
        print("-"*70)
        results['variant_analysis'] = self.analyze_variant_position_states(num_samples=50)

        # 3. Sequence evolution
        print("\n" + "-"*70)
        print("Step 3/3: Visualizing sequence position evolution")
        print("-"*70)
        self.analyze_sequence_position_evolution(num_samples=3)

        # Save report
        self._save_report(results)

        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        print(f"Results saved to: {self.output_dir}")

        return results

    def _save_report(self, results: dict):
        """Save text summary report"""
        report_path = self.output_dir / 'hidden_state_report.txt'

        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("HIDDEN STATE ANALYSIS REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("-"*70 + "\n")
            f.write("CLS REPRESENTATION SUMMARY\n")
            f.write("-"*70 + "\n")
            cls_results = results['cls_analysis']
            f.write(f"Between-class distance: {cls_results['between_class_dist']:.3f}\n")
            f.write(f"Within-class 0 distance: {cls_results['within_class_dists'][0]:.3f}\n")
            f.write(f"Within-class 1 distance: {cls_results['within_class_dists'][1]:.3f}\n")

            sep_ratio = (cls_results['between_class_dist'] /
                        (sum(cls_results['within_class_dists']) / 2))
            f.write(f"Separation ratio: {sep_ratio:.3f}\n")

            f.write("\n" + "-"*70 + "\n")
            f.write("VARIANT POSITION ANALYSIS\n")
            f.write("-"*70 + "\n")
            var_results = results['variant_analysis']
            p_vals = var_results['p_values']
            effects = var_results['effect_sizes']

            f.write(f"Significant dimensions (p < 0.01): {(p_vals < 0.01).sum()}/{len(p_vals)}\n")
            f.write(f"Significant dimensions (p < 0.05): {(p_vals < 0.05).sum()}/{len(p_vals)}\n")
            f.write(f"Mean absolute effect size: {np.abs(effects).mean():.4f}\n\n")

            f.write("Top 10 most discriminative dimensions:\n")
            for rank, dim in enumerate(var_results['top_discriminative_dims'][:10], 1):
                f.write(f"  {rank:2d}. Dim {dim:3d}: p={p_vals[dim]:.2e}, ")
                f.write(f"effect_size={effects[dim]:+.4f}\n")

        print(f"Saved report: hidden_state_report.txt")
