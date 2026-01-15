"""
Mechanistic Attention Analysis Module

Provides MechanisticAttentionAnalyzer class for analyzing attention patterns
in genomic models with focus on variant positions and mechanistic interpretability.
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
from data.generic_data_loader import get_data_loader
from models.load_model import load_base_model

class MechanisticAttentionAnalyzer:
    """Mechanistic interpretability-focused attention analysis"""

    def __init__(self,
                 dataset_name: str,
                 num_samples: int = 100,
                 model_name: str = "zhihan1996/DNABERT-2-117M",
                 output_dir: str = None):
        """
        Initialize analyzer

        Args:
            dataset_name: Name of dataset (eqtl, sqtl, clinvar, etc.)
            num_samples: Number of samples to analyze
            model_name: Model to analyze
            output_dir: Output directory
        """
        self.dataset_name = dataset_name
        self.num_samples = num_samples
        self.model_name = model_name

        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"outputs/{dataset_name}_mechanistic_attention/{timestamp}"

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load model
        print("\nLoading model...")
        self.model, self.tokenizer, self.config = load_base_model(
            model_name, device=str(self.device)
        )

        print(f"\nModel configuration:")
        print(f"  Layers: {self.config.num_hidden_layers}")
        print(f"  Attention heads: {self.config.num_attention_heads}")
        print(f"  Hidden size: {self.config.hidden_size}")

        # Load data
        print(f"\nLoading {dataset_name} data...")
        self.data_loader = get_data_loader(dataset_name, num_records=num_samples, seq_length=1024)
        self.samples = self.data_loader.load_data()
        self.data_loader.print_statistics()

        # Separate by class
        self.class_0_samples = [s for s in self.samples if s.label == 0]
        self.class_1_samples = [s for s in self.samples if s.label == 1]

        print(f"\nClass 0: {len(self.class_0_samples)} samples")
        print(f"Class 1: {len(self.class_1_samples)} samples")

    def get_variant_token_position(self, sequence: str, variant_bp_pos: int, max_length: int = 512) -> int:
        """
        Get the correct token position for a variant after tokenization.

        Accounts for:
        - Special tokens ([CLS], [SEP], [PAD])
        - k-mer tokenization in DNABERT-2

        Args:
            sequence: DNA sequence
            variant_bp_pos: Base pair position of variant
            max_length: Max tokenization length

        Returns:
            Token index corresponding to variant position
        """
        # Tokenize
        tokens = self.tokenizer.tokenize(sequence[:max_length])

        # DNABERT-2 uses 6-mer tokenization by default
        # Each token covers ~6 base pairs
        # CLS token is position 0

        # Account for CLS token
        kmer_size = 6
        approximate_token_pos = 1 + (variant_bp_pos // kmer_size)

        # Bound check
        approximate_token_pos = min(approximate_token_pos, len(tokens) - 1)

        return approximate_token_pos

    def compute_attention_with_metadata(self, sequence: str, variant_pos: int = None, max_length: int = 512):
        """
        Compute attention with proper variant position tracking

        Returns:
            Dictionary with:
            - attention: (layers, heads, seq_len, seq_len)
            - tokens: List of token strings
            - variant_token_pos: Token index of variant (if provided)
            - cls_token_pos: Token index of CLS token (always 0)
        """
        # Tokenize
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding="max_length"
        ).to(self.device)

        # Get attention
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        attentions = outputs.attentions
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        # Stack: (layers, heads, seq_len, seq_len)
        attention_weights = torch.stack([attn[0] for attn in attentions]).cpu().numpy()

        result = {
            'attention': attention_weights,
            'tokens': tokens,
            'cls_token_pos': 0,  # CLS is always first
            'attention_mask': inputs['attention_mask'][0].cpu().numpy()
        }

        if variant_pos is not None:
            result['variant_token_pos'] = self.get_variant_token_position(
                sequence, variant_pos, max_length
            )

        return result

    def compute_attention_entropy(self, attention: np.ndarray) -> float:
        """
        Compute entropy of attention distribution (measure of focus vs diffusion)

        Args:
            attention: Attention weights for one head (seq_len,)

        Returns:
            Entropy value (higher = more diffuse)
        """
        # Add small epsilon to avoid log(0)
        attention = attention + 1e-10
        attention = attention / attention.sum()
        return -np.sum(attention * np.log(attention))

    def analyze_per_head_behavior(self, num_samples: int = 50):
        """
        Analyze each attention head's behavior across layers
        Identifies heads with:
        - High entropy (diffuse attention)
        - Low entropy (focused attention)
        - High variance across positions (structured patterns)

        Note: Currently analyzes CLS token attention. For variant-specific tasks,
        consider also computing metrics using variant token as query.

        CONNECTION TO ABLATION/PATCHING:
        Heads identified here (low entropy + high significance in variant analysis)
        can be directly tested via:
        1. Ablation: Zero out these heads' outputs and measure classification impact
        2. Patching: Swap these heads between significant/non-significant examples
        3. Circuit analysis: Test if these heads form functional circuits

        Use the returned (layer, head) indices with run_activation_patching.py
        or run_circuit_analysis.py to establish causal importance.

        Returns head importance scores
        """
        print("\n" + "="*70)
        print("PER-HEAD ATTENTION BEHAVIOR ANALYSIS")
        print("="*70)

        n_layers = self.config.num_hidden_layers
        n_heads = self.config.num_attention_heads

        # Track metrics per head
        head_entropy = np.zeros((n_layers, n_heads))
        head_variance = np.zeros((n_layers, n_heads))
        head_max_attention = np.zeros((n_layers, n_heads))

        samples_to_analyze = self.samples[:num_samples]

        print(f"Analyzing {len(samples_to_analyze)} samples across {n_layers} layers and {n_heads} heads...")

        for sample in tqdm(samples_to_analyze):
            result = self.compute_attention_with_metadata(
                sample.ref_sequence,
                sample.variant_pos
            )
            attention = result['attention']  # (layers, heads, seq_len, seq_len)

            for layer in range(n_layers):
                for head in range(n_heads):
                    # Get attention from CLS token
                    cls_attention = attention[layer, head, 0, :]

                    head_entropy[layer, head] += self.compute_attention_entropy(cls_attention)
                    head_variance[layer, head] += np.var(cls_attention)
                    head_max_attention[layer, head] += np.max(cls_attention)

        # Average metrics
        head_entropy /= len(samples_to_analyze)
        head_variance /= len(samples_to_analyze)
        head_max_attention /= len(samples_to_analyze)

        # Plot heatmaps
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        im1 = axes[0].imshow(head_entropy, aspect='auto', cmap='viridis')
        axes[0].set_title('Attention Entropy per Head\n(Higher = more diffuse)', fontweight='bold')
        axes[0].set_xlabel('Head Index')
        axes[0].set_ylabel('Layer Index')
        plt.colorbar(im1, ax=axes[0])

        im2 = axes[1].imshow(head_variance, aspect='auto', cmap='plasma')
        axes[1].set_title('Attention Variance per Head\n(Higher = more structured)', fontweight='bold')
        axes[1].set_xlabel('Head Index')
        axes[1].set_ylabel('Layer Index')
        plt.colorbar(im2, ax=axes[1])

        im3 = axes[2].imshow(head_max_attention, aspect='auto', cmap='coolwarm')
        axes[2].set_title('Max Attention per Head\n(Peak focus strength)', fontweight='bold')
        axes[2].set_xlabel('Head Index')
        axes[2].set_ylabel('Layer Index')
        plt.colorbar(im3, ax=axes[2])

        plt.tight_layout()
        plt.savefig(self.output_dir / '1_per_head_behavior.png', dpi=150, bbox_inches='tight')
        print(f"Saved: 1_per_head_behavior.png")

        # Identify interesting heads
        print("\nTop 5 most focused heads (low entropy):")
        top_focused = np.argsort(head_entropy.flatten())[:5]
        for idx in top_focused:
            layer = idx // n_heads
            head = idx % n_heads
            print(f"  Layer {layer}, Head {head}: entropy = {head_entropy[layer, head]:.3f}")

        print("\nTop 5 most structured heads (high variance):")
        top_structured = np.argsort(head_variance.flatten())[::-1][:5]
        for idx in top_structured:
            layer = idx // n_heads
            head = idx % n_heads
            print(f"  Layer {layer}, Head {head}: variance = {head_variance[layer, head]:.4f}")

        return {
            'entropy': head_entropy,
            'variance': head_variance,
            'max_attention': head_max_attention,
            'top_focused_heads': top_focused,
            'top_structured_heads': top_structured
        }

    def analyze_variant_specific_attention(self, query_type: str = 'cls', num_samples: int = None):
        """
        Analyze attention FROM specific query tokens TO variant positions

        Args:
            query_type: 'cls' or 'variant' - which token to use as query
            num_samples: Number of samples per class

        Returns:
            Statistical comparison between classes
        """
        print("\n" + "="*70)
        print(f"VARIANT-SPECIFIC ATTENTION ANALYSIS (query: {query_type.upper()})")
        print("="*70)

        # Use all available samples for maximum statistical power
        if num_samples is None:
            num_samples = min(len(self.class_0_samples), len(self.class_1_samples))

        class_0_attentions = []  # Shape will be (n_valid_samples, n_layers, n_heads)
        class_1_attentions = []

        # Track actual samples processed (some may be skipped if variant_pos < 0)
        actual_counts = [0, 0]

        print(f"\nAnalyzing {num_samples} samples per class...")

        for class_idx, (samples, storage) in enumerate([
            (self.class_0_samples[:num_samples], class_0_attentions),
            (self.class_1_samples[:num_samples], class_1_attentions)
        ]):
            for sample in tqdm(samples):
                if sample.variant_pos < 0:
                    continue

                actual_counts[class_idx] += 1

                result = self.compute_attention_with_metadata(
                    sample.ref_sequence,
                    sample.variant_pos
                )

                attention = result['attention']  # (layers, heads, seq_len, seq_len)
                variant_token = result['variant_token_pos']

                if query_type == 'cls':
                    query_pos = 0
                else:  # variant
                    query_pos = variant_token

                # Attention FROM query TO variant position
                # Shape: (layers, heads)
                attn_to_variant = attention[:, :, query_pos, variant_token]

                storage.append(attn_to_variant)

        # Convert to arrays
        class_0_attentions = np.array(class_0_attentions)  # (n_valid_samples, layers, heads)
        class_1_attentions = np.array(class_1_attentions)

        print(f"\nActual samples analyzed: Class 0={actual_counts[0]}, Class 1={actual_counts[1]}")
        print(f"(Skipped {num_samples - actual_counts[0]} + {num_samples - actual_counts[1]} samples with invalid variant positions)")

        # Statistical testing per layer/head
        # Note: Using independent t-tests assuming normality.
        # For small samples or non-normal distributions, consider Mann-Whitney U test.
        n_layers, n_heads = class_0_attentions.shape[1], class_0_attentions.shape[2]
        p_values = np.zeros((n_layers, n_heads))
        effect_sizes = np.zeros((n_layers, n_heads))

        for layer in range(n_layers):
            for head in range(n_heads):
                class_0_vals = class_0_attentions[:, layer, head]
                class_1_vals = class_1_attentions[:, layer, head]

                # T-test
                t_stat, p_val = stats.ttest_ind(class_0_vals, class_1_vals)
                p_values[layer, head] = p_val

                # Cohen's d (effect size)
                pooled_std = np.sqrt((np.var(class_0_vals) + np.var(class_1_vals)) / 2)
                effect_sizes[layer, head] = (np.mean(class_0_vals) - np.mean(class_1_vals)) / (pooled_std + 1e-10)

        # Plot results
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # P-values heatmap
        im1 = axes[0].imshow(-np.log10(p_values + 1e-10), aspect='auto', cmap='Reds')
        axes[0].set_title(f'Statistical Significance\n-log10(p-value) from query={query_type.upper()} to variant',
                          fontweight='bold')
        axes[0].set_xlabel('Head Index')
        axes[0].set_ylabel('Layer Index')
        # Note: -log10(0.05) ≈ 1.3, -log10(0.01) ≈ 2.0 (shown in colorbar)
        plt.colorbar(im1, ax=axes[0], label='-log10(p)')

        # Effect sizes
        im2 = axes[1].imshow(effect_sizes, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
        axes[1].set_title(f"Effect Size (Cohen's d)\nClass 0 vs Class 1", fontweight='bold')
        axes[1].set_xlabel('Head Index')
        axes[1].set_ylabel('Layer Index')
        plt.colorbar(im2, ax=axes[1], label="Effect size")

        plt.tight_layout()
        plt.savefig(self.output_dir / f'2_variant_attention_stats_{query_type}.png',
                    dpi=150, bbox_inches='tight')
        print(f"Saved: 2_variant_attention_stats_{query_type}.png")

        # Find most significant heads
        significant_mask = p_values < 0.01
        print(f"\nHeads with significant difference (p < 0.01): {significant_mask.sum()}/{n_layers * n_heads}")

        if significant_mask.sum() > 0:
            top_significant = np.argsort(p_values.flatten())[:5]
            print("\nTop 5 most significant heads:")
            for idx in top_significant:
                layer = idx // n_heads
                head = idx % n_heads
                print(f"  Layer {layer}, Head {head}: p={p_values[layer, head]:.2e}, effect_size={effect_sizes[layer, head]:.3f}")

        return {
            'p_values': p_values,
            'effect_sizes': effect_sizes,
            'class_0_attentions': class_0_attentions,
            'class_1_attentions': class_1_attentions
        }

    def visualize_specific_head_attention(self, layer: int, head: int, num_samples: int = 3):
        """
        Visualize attention patterns for a specific head across sample examples
        Shows how attention relates to sequence basepairs and output labels

        Args:
            layer: Layer index
            head: Head index
            num_samples: Number of examples per class
        """
        print(f"\n" + "="*70)
        print(f"VISUALIZING LAYER {layer}, HEAD {head}")
        print("="*70)

        fig, axes = plt.subplots(2, num_samples, figsize=(6*num_samples, 8))
        if num_samples == 1:
            axes = axes.reshape(-1, 1)

        kmer_size = 6  # DNABERT uses 6-mer tokenization

        for class_idx, (samples, class_name) in enumerate([
            (self.class_0_samples, self.data_loader.get_label_name(0)),
            (self.class_1_samples, self.data_loader.get_label_name(1))
        ]):
            for sample_idx in range(min(num_samples, len(samples))):
                sample = samples[sample_idx]

                result = self.compute_attention_with_metadata(
                    sample.ref_sequence,
                    sample.variant_pos
                )

                attention = result['attention'][layer, head]  # (seq_len, seq_len)
                variant_token = result.get('variant_token_pos', -1)

                # Convert variant token position to approximate basepair position
                variant_bp = (variant_token - 1) * kmer_size if variant_token > 0 else sample.variant_pos

                ax = axes[class_idx, sample_idx]

                # Use log scale for better visualization of small attention values
                # Add small epsilon to avoid log(0)
                attention_vis = np.log10(attention + 1e-10)

                # Plot full attention matrix with log scale
                im = ax.imshow(attention_vis, cmap='viridis', aspect='auto')

                # Mark variant position with red dashed lines
                if variant_token >= 0:
                    ax.axvline(variant_token, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Variant')
                    ax.axhline(variant_token, color='red', linestyle='--', linewidth=2, alpha=0.7)

                ax.set_title(f'{class_name}\nSample {sample_idx+1}\n(variant at token {variant_token} ≈ {variant_bp}bp)',
                            fontweight='bold', fontsize=10)
                ax.set_xlabel('Key Position')
                ax.set_ylabel('Query Position')

                # Add colorbar with log-scale attention values
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('log₁₀(Attention)', rotation=270, labelpad=15, fontsize=8)

                # Add legend for variant marker
                if variant_token >= 0 and sample_idx == 0:
                    ax.legend(loc='upper right', fontsize=8)

        plt.suptitle(f'Attention Patterns: Layer {layer}, Head {head}\nHow Attention Fires Across Sequence for Each Class',
                    fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'3_head_L{layer}H{head}_examples.png',
                    dpi=150, bbox_inches='tight')
        print(f"Saved: 3_head_L{layer}H{head}_examples.png")

    def visualize_position_wise_attention(self, layer: int, head: int,
                                          num_samples: int = 50,
                                          num_examples: int = 5):
        """
        Visualize how attention activates across sequence positions.

        Creates plots showing:
        1. Averaged attention profile (mean ± std) by class
        2. Individual example overlays for a few samples

        Args:
            layer: Layer index
            head: Head index
            num_samples: Number of samples to use for averaging
            num_examples: Number of individual examples to overlay
        """
        print(f"\nGenerating position-wise attention plots for Layer {layer}, Head {head}...")

        # Determine k-mer size for DNABERT models
        kmer_size = 6 if 'DNA_bert_6' in self.model_name else 1

        # Collect attention patterns FROM CLS token TO all positions
        class_0_patterns = []  # (samples, seq_len)
        class_1_patterns = []
        class_0_variant_pos = []
        class_1_variant_pos = []

        # Class 0 samples
        for sample in self.class_0_samples[:num_samples]:
            result = self.compute_attention_with_metadata(sample.ref_sequence, sample.variant_pos)
            attention = result['attention'][layer, head]  # (seq_len, seq_len)
            cls_attention = attention[0, :]  # FROM CLS TO all positions
            class_0_patterns.append(cls_attention)
            class_0_variant_pos.append(result.get('variant_token_pos', -1))

        # Class 1 samples
        for sample in self.class_1_samples[:num_samples]:
            result = self.compute_attention_with_metadata(sample.ref_sequence, sample.variant_pos)
            attention = result['attention'][layer, head]
            cls_attention = attention[0, :]
            class_1_patterns.append(cls_attention)
            class_1_variant_pos.append(result.get('variant_token_pos', -1))

        # Convert to numpy arrays
        class_0_patterns = np.array(class_0_patterns)  # (num_samples, seq_len)
        class_1_patterns = np.array(class_1_patterns)

        # Calculate statistics
        class_0_mean = np.mean(class_0_patterns, axis=0)
        class_0_std = np.std(class_0_patterns, axis=0)
        class_1_mean = np.mean(class_1_patterns, axis=0)
        class_1_std = np.std(class_1_patterns, axis=0)

        # Convert token positions to basepair positions
        seq_len = class_0_patterns.shape[1]
        token_positions = np.arange(seq_len)
        bp_positions = token_positions * kmer_size  # Approximate basepair positions

        # Get class names
        class_0_name = self.data_loader.get_label_name(0)
        class_1_name = self.data_loader.get_label_name(1)

        # Average variant position in basepairs
        avg_variant_bp_0 = np.mean([pos * kmer_size for pos in class_0_variant_pos if pos > 0])
        avg_variant_bp_1 = np.mean([pos * kmer_size for pos in class_1_variant_pos if pos > 0])

        # Create figure with 2 subplots
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))

        # ============================================================
        # TOP PANEL: Averaged attention profiles
        # ============================================================
        ax = axes[0]

        # Plot mean with shaded std
        ax.plot(bp_positions, class_0_mean, label=f'{class_0_name} (mean)',
                color='#3498db', linewidth=2.5, alpha=0.9)
        ax.fill_between(bp_positions,
                        class_0_mean - class_0_std,
                        class_0_mean + class_0_std,
                        alpha=0.25, color='#3498db', label=f'{class_0_name} (±1 std)')

        ax.plot(bp_positions, class_1_mean, label=f'{class_1_name} (mean)',
                color='#e74c3c', linewidth=2.5, alpha=0.9)
        ax.fill_between(bp_positions,
                        class_1_mean - class_1_std,
                        class_1_mean + class_1_std,
                        alpha=0.25, color='#e74c3c', label=f'{class_1_name} (±1 std)')

        # Mark average variant positions
        if avg_variant_bp_0 > 0:
            ax.axvline(avg_variant_bp_0, color='#3498db', linestyle='--',
                      linewidth=2, alpha=0.6, label=f'{class_0_name} variant')
        if avg_variant_bp_1 > 0:
            ax.axvline(avg_variant_bp_1, color='#e74c3c', linestyle='--',
                      linewidth=2, alpha=0.6, label=f'{class_1_name} variant')

        ax.set_xlabel('Sequence Position (bp)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Attention Weight', fontsize=12, fontweight='bold')
        ax.set_title(f'Layer {layer}, Head {head}: Averaged Attention from [CLS] Token Across Sequence\n'
                    f'(n={num_samples} samples per class)',
                    fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # ============================================================
        # BOTTOM PANEL: Individual examples
        # ============================================================
        ax = axes[1]

        # Plot individual examples (first num_examples from each class)
        for i in range(min(num_examples, len(class_0_patterns))):
            ax.plot(bp_positions, class_0_patterns[i],
                   color='#3498db', alpha=0.4, linewidth=1.5)

        for i in range(min(num_examples, len(class_1_patterns))):
            ax.plot(bp_positions, class_1_patterns[i],
                   color='#e74c3c', alpha=0.4, linewidth=1.5)

        # Add dummy lines for legend
        ax.plot([], [], color='#3498db', alpha=0.6, linewidth=2,
               label=f'{class_0_name} (n={min(num_examples, len(class_0_patterns))} examples)')
        ax.plot([], [], color='#e74c3c', alpha=0.6, linewidth=2,
               label=f'{class_1_name} (n={min(num_examples, len(class_1_patterns))} examples)')

        # Mark average variant positions
        if avg_variant_bp_0 > 0:
            ax.axvline(avg_variant_bp_0, color='#3498db', linestyle='--',
                      linewidth=2, alpha=0.5)
        if avg_variant_bp_1 > 0:
            ax.axvline(avg_variant_bp_1, color='#e74c3c', linestyle='--',
                      linewidth=2, alpha=0.5)

        ax.set_xlabel('Sequence Position (bp)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Attention Weight', fontsize=12, fontweight='bold')
        ax.set_title(f'Individual Example Attention Patterns\n'
                    f'(Each line = one sample, colors = class labels)',
                    fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / f'4_position_attention_L{layer}H{head}.png',
                    dpi=150, bbox_inches='tight')
        print(f"Saved: 4_position_attention_L{layer}H{head}.png")
        plt.close()

        # Also create a focused view on variant region (±200bp window)
        if avg_variant_bp_0 > 0 or avg_variant_bp_1 > 0:
            self._visualize_variant_region_zoom(
                layer, head,
                class_0_patterns, class_1_patterns,
                bp_positions, avg_variant_bp_0, avg_variant_bp_1,
                class_0_name, class_1_name,
                num_examples
            )

    def _visualize_variant_region_zoom(self, layer: int, head: int,
                                       class_0_patterns: np.ndarray,
                                       class_1_patterns: np.ndarray,
                                       bp_positions: np.ndarray,
                                       avg_variant_bp_0: float,
                                       avg_variant_bp_1: float,
                                       class_0_name: str,
                                       class_1_name: str,
                                       num_examples: int):
        """Create zoomed-in view of attention around variant region"""

        # Determine variant center (use average of both classes)
        variant_center = (avg_variant_bp_0 + avg_variant_bp_1) / 2 if avg_variant_bp_0 > 0 and avg_variant_bp_1 > 0 else max(avg_variant_bp_0, avg_variant_bp_1)

        # Define window (±200bp around variant)
        window = 200
        start_bp = max(0, variant_center - window)
        end_bp = variant_center + window

        # Find corresponding indices
        start_idx = np.searchsorted(bp_positions, start_bp)
        end_idx = np.searchsorted(bp_positions, end_bp)

        # Extract window
        bp_window = bp_positions[start_idx:end_idx]
        class_0_window = class_0_patterns[:, start_idx:end_idx]
        class_1_window = class_1_patterns[:, start_idx:end_idx]

        if len(bp_window) == 0:
            return  # Skip if window is empty

        # Calculate statistics
        class_0_mean = np.mean(class_0_window, axis=0)
        class_0_std = np.std(class_0_window, axis=0)
        class_1_mean = np.mean(class_1_window, axis=0)
        class_1_std = np.std(class_1_window, axis=0)

        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # TOP: Averaged
        ax = axes[0]
        ax.plot(bp_window, class_0_mean, label=f'{class_0_name} (mean)',
                color='#3498db', linewidth=3, alpha=0.9)
        ax.fill_between(bp_window,
                        class_0_mean - class_0_std,
                        class_0_mean + class_0_std,
                        alpha=0.3, color='#3498db', label=f'{class_0_name} (±1 std)')

        ax.plot(bp_window, class_1_mean, label=f'{class_1_name} (mean)',
                color='#e74c3c', linewidth=3, alpha=0.9)
        ax.fill_between(bp_window,
                        class_1_mean - class_1_std,
                        class_1_mean + class_1_std,
                        alpha=0.3, color='#e74c3c', label=f'{class_1_name} (±1 std)')

        # Mark variant position
        ax.axvline(variant_center, color='black', linestyle='--',
                  linewidth=2.5, alpha=0.7, label='Variant position')

        ax.set_xlabel('Sequence Position (bp)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Attention Weight', fontsize=12, fontweight='bold')
        ax.set_title(f'Layer {layer}, Head {head}: Zoomed View Around Variant (±{window}bp)\n'
                    f'Averaged Attention from [CLS] Token',
                    fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # BOTTOM: Individual examples
        ax = axes[1]
        for i in range(min(num_examples, len(class_0_window))):
            ax.plot(bp_window, class_0_window[i],
                   color='#3498db', alpha=0.5, linewidth=1.5)

        for i in range(min(num_examples, len(class_1_window))):
            ax.plot(bp_window, class_1_window[i],
                   color='#e74c3c', alpha=0.5, linewidth=1.5)

        # Dummy for legend
        ax.plot([], [], color='#3498db', alpha=0.7, linewidth=2,
               label=f'{class_0_name} examples')
        ax.plot([], [], color='#e74c3c', alpha=0.7, linewidth=2,
               label=f'{class_1_name} examples')

        ax.axvline(variant_center, color='black', linestyle='--',
                  linewidth=2.5, alpha=0.7)

        ax.set_xlabel('Sequence Position (bp)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Attention Weight', fontsize=12, fontweight='bold')
        ax.set_title(f'Individual Examples (±{window}bp window)',
                    fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / f'5_variant_zoom_L{layer}H{head}.png',
                    dpi=150, bbox_inches='tight')
        print(f"Saved: 5_variant_zoom_L{layer}H{head}.png")
        plt.close()

    def run_full_mechanistic_analysis(self):
        """Run complete mechanistic attention analysis"""
        print("\n" + "="*70)
        print("MECHANISTIC ATTENTION ANALYSIS")
        print("="*70)
        print(f"Dataset: {self.dataset_name}")
        print(f"Samples: {self.num_samples}")
        print(f"Output: {self.output_dir}")
        print("="*70)

        results = {}

        # 1. Per-head behavior analysis
        print("\n" + "-"*70)
        print("Step 1/4: Analyzing per-head behavior")
        print("-"*70)
        results['head_behavior'] = self.analyze_per_head_behavior(num_samples=50)

        # 2. Variant-specific attention (CLS query)
        print("\n" + "-"*70)
        print("Step 2/4: Analyzing CLS-to-variant attention")
        print("-"*70)
        results['cls_to_variant'] = self.analyze_variant_specific_attention(
            query_type='cls', num_samples=50
        )

        # 3. Variant-specific attention (variant query)
        print("\n" + "-"*70)
        print("Step 3/4: Analyzing variant self-attention")
        print("-"*70)
        results['variant_self'] = self.analyze_variant_specific_attention(
            query_type='variant', num_samples=50
        )

        # 4. Visualize most significant head
        print("\n" + "-"*70)
        print("Step 4/5: Visualizing most significant heads")
        print("-"*70)

        # Find most significant head from CLS analysis
        p_values = results['cls_to_variant']['p_values']
        most_sig_idx = np.argmin(p_values)
        n_heads = self.config.num_attention_heads
        best_layer = most_sig_idx // n_heads
        best_head = most_sig_idx % n_heads

        print(f"Most significant head: Layer {best_layer}, Head {best_head}")
        self.visualize_specific_head_attention(best_layer, best_head, num_samples=3)

        # 5. Position-wise attention analysis
        print("\n" + "-"*70)
        print("Step 5/5: Analyzing position-wise attention activation")
        print("-"*70)

        # Generate position-wise plots for top 3 most significant heads
        top_heads_indices = np.argsort(p_values.flatten())[:3]
        for idx in top_heads_indices:
            layer = idx // n_heads
            head = idx % n_heads
            print(f"\nGenerating position plots for Layer {layer}, Head {head} (p={p_values[layer, head]:.2e})...")
            self.visualize_position_wise_attention(layer, head, num_samples=50, num_examples=5)

        # Save summary report
        self._save_report(results)

        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        print(f"Results saved to: {self.output_dir}")

        return results

    def _save_report(self, results: dict):
        """Save text summary report"""
        report_path = self.output_dir / 'mechanistic_attention_report.txt'

        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("MECHANISTIC ATTENTION ANALYSIS REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Samples analyzed: {self.num_samples}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("-"*70 + "\n")
            f.write("HEAD BEHAVIOR SUMMARY\n")
            f.write("-"*70 + "\n")

            entropy = results['head_behavior']['entropy']
            variance = results['head_behavior']['variance']
            n_heads = self.config.num_attention_heads

            f.write(f"\nMost focused heads (lowest entropy):\n")
            for idx in results['head_behavior']['top_focused_heads']:
                layer = idx // n_heads
                head = idx % n_heads
                f.write(f"  Layer {layer:2d}, Head {head:2d}: entropy = {entropy[layer, head]:.4f}\n")

            f.write(f"\nMost structured heads (highest variance):\n")
            for idx in results['head_behavior']['top_structured_heads'][:5]:
                layer = idx // n_heads
                head = idx % n_heads
                f.write(f"  Layer {layer:2d}, Head {head:2d}: variance = {variance[layer, head]:.6f}\n")

            f.write("\n" + "-"*70 + "\n")
            f.write("VARIANT ATTENTION STATISTICS (CLS query)\n")
            f.write("-"*70 + "\n")

            p_vals = results['cls_to_variant']['p_values']
            effects = results['cls_to_variant']['effect_sizes']
            sig_mask = p_vals < 0.01

            f.write(f"\nSignificant heads (p < 0.01): {sig_mask.sum()}/{p_vals.size}\n")
            f.write(f"Significant heads (p < 0.05): {(p_vals < 0.05).sum()}/{p_vals.size}\n\n")

            f.write("Top 10 most discriminative heads:\n")
            top_idx = np.argsort(p_vals.flatten())[:10]
            for rank, idx in enumerate(top_idx, 1):
                layer = idx // n_heads
                head = idx % n_heads
                f.write(f"  {rank:2d}. Layer {layer:2d}, Head {head:2d}: ")
                f.write(f"p={p_vals[layer, head]:.2e}, effect_size={effects[layer, head]:+.4f}\n")

        print(f"Saved report: mechanistic_attention_report.txt")

    def get_top_heads_for_ablation(self, results: dict, top_k: int = 10) -> List[Tuple[int, int]]:
        """
        Extract top (layer, head) pairs for follow-up ablation/patching experiments.

        Combines:
        - Low entropy (focused heads)
        - High statistical significance (p < 0.05)
        - Large effect sizes

        Args:
            results: Results dict from run_full_mechanistic_analysis()
            top_k: Number of top heads to return

        Returns:
            List of (layer, head) tuples ranked by importance
        """
        entropy = results['head_behavior']['entropy']
        p_values = results['cls_to_variant']['p_values']
        effect_sizes = results['cls_to_variant']['effect_sizes']

        n_heads = self.config.num_attention_heads

        # Score each head: combine low entropy, low p-value, high effect size
        scores = []
        for layer in range(self.config.num_hidden_layers):
            for head in range(n_heads):
                # Normalize metrics to [0, 1] range
                entropy_score = 1 - (entropy[layer, head] / entropy.max())
                p_score = 1 - (p_values[layer, head] / (p_values.max() + 1e-10))
                effect_score = abs(effect_sizes[layer, head])

                # Combined score (equal weighting)
                combined = (entropy_score + p_score + effect_score) / 3
                scores.append((combined, layer, head))

        # Sort by combined score
        scores.sort(reverse=True)

        return [(layer, head) for _, layer, head in scores[:top_k]]


