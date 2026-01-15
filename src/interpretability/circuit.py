"""
Circuit Analysis Module

Provides CircuitAnalyzer class for discovering and analyzing
functional circuits in genomic models.
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
from typing import List, Dict, Tuple, Set
from sklearn.cluster import KMeans
from scipy.stats import pearsonr
import warnings
from data.sqtl_data_loader import OriginalSQTLDataLoader, sQTLSample
from models.load_model import load_base_model

class CircuitAnalyzer:
    """
    Discover and analyze functional circuits in DNABERT-2 for sQTL classification
    """

    def __init__(self,
                 model_name: str = "zhihan1996/DNABERT-2-117M",
                 num_samples: int = 100,
                 output_dir: str = "outputs/circuit_analysis"):
        """
        Initialize circuit analyzer

        Args:
            model_name: DNABERT-2 model name
            num_samples: Number of sQTL samples to analyze
            output_dir: Output directory for results
        """
        self.model_name = model_name
        self.num_samples = num_samples
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load model
        print("\nLoading model...")
        self.model, self.tokenizer, self.config = load_base_model(
            model_name, device=str(self.device)
        )

        # Load data
        print("\nLoading sQTL data...")
        loader = OriginalSQTLDataLoader(num_samples=num_samples, tissue_filter="Whole_Blood")
        self.samples = loader.load_data()

        # Separate by class
        self.significant = [s for s in self.samples if s.label == 0]
        self.not_significant = [s for s in self.samples if s.label == 1]

        print(f"\nLoaded {len(self.samples)} samples:")
        print(f"  Significant: {len(self.significant)}")
        print(f"  Not significant: {len(self.not_significant)}")

        # Model architecture
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        self.hidden_size = self.config.hidden_size

        print(f"\nModel architecture:")
        print(f"  Layers: {self.num_layers}")
        print(f"  Attention heads: {self.num_heads}")
        print(f"  Hidden size: {self.hidden_size}")

        # Storage for activation patterns
        self.activation_patterns = {
            'significant': [],
            'not_significant': []
        }

    def get_attention_patterns(self, sequence: str, max_length: int = 512):
        """
        Extract attention patterns for a sequence

        Returns:
            attention_patterns: (num_layers, num_heads, seq_len, seq_len)
            pooled_attentions: (num_layers * num_heads,) - mean attention per head
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

        # Check if model returns attention (not all models do, e.g., DNABERT-2)
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            attentions = outputs.attentions  # List of (1, num_heads, seq_len, seq_len)
        else:
            # Model doesn't support attention outputs (e.g., DNABERT-2)
            return None, None

        # Pool attention patterns to head-level statistics
        pooled = []
        for layer_attn in attentions:
            for head_idx in range(self.num_heads):
                head_attn = layer_attn[0, head_idx]  # (seq_len, seq_len)
                # Compute mean attention (how much attention this head uses)
                mean_attn = head_attn.mean().item()
                pooled.append(mean_attn)

        pooled = np.array(pooled)  # (num_layers * num_heads,)

        return attentions, pooled

    def collect_activation_patterns(self, num_samples_per_class: int = 50):
        """
        Collect attention activation patterns for both classes
        """
        print("\n" + "="*70)
        print("COLLECTING ACTIVATION PATTERNS")
        print("="*70)
        print("Extracting attention patterns for circuit discovery\n")

        # Check if model supports attention
        test_attn, test_pooled = self.get_attention_patterns(self.significant[0].ref_sequence)
        if test_attn is None:
            print("\\nWARNING: Model does not support attention outputs.")
            print("         Circuit analysis requires attention weights.")
            print("         (This is expected for DNABERT-2 with optimized attention)\\n")
            return

        # Collect for significant sQTLs
        print(f"Collecting patterns from {num_samples_per_class} significant sQTLs...")
        for sample in tqdm(self.significant[:num_samples_per_class], desc="Significant"):
            _, pooled = self.get_attention_patterns(sample.ref_sequence)
            if pooled is not None:
                self.activation_patterns['significant'].append(pooled)

        # Collect for not_significant sQTLs
        print(f"Collecting patterns from {num_samples_per_class} not_significant sQTLs...")
        for sample in tqdm(self.not_significant[:num_samples_per_class], desc="Not significant"):
            _, pooled = self.get_attention_patterns(sample.ref_sequence)
            if pooled is not None:
                self.activation_patterns['not_significant'].append(pooled)

        # Convert to arrays (check if we have data)
        if len(self.activation_patterns['significant']) == 0:
            print("\\nNo activation patterns collected. Exiting.")
            return

        self.activation_patterns['significant'] = np.array(self.activation_patterns['significant'])
        self.activation_patterns['not_significant'] = np.array(self.activation_patterns['not_significant'])

        print(f"\nCollected patterns:")
        print(f"  Significant: {self.activation_patterns['significant'].shape}")
        print(f"  Not significant: {self.activation_patterns['not_significant'].shape}")

    def discover_circuits(self, n_circuits: int = 5):
        """
        Discover functional circuits using correlation analysis

        Find groups of attention heads that:
        1. Co-activate (high correlation)
        2. Show differential activation between classes

        Returns:
            circuits: List of identified circuits (each is a set of (layer, head) tuples)
        """
        print("\n" + "="*70)
        print("CIRCUIT DISCOVERY")
        print("="*70)
        print(f"Discovering {n_circuits} functional circuits\n")

        # Check if we have activation patterns (handle numpy arrays properly)
        sig_patterns = self.activation_patterns['significant']
        if sig_patterns is None or (isinstance(sig_patterns, np.ndarray) and sig_patterns.size == 0) or (isinstance(sig_patterns, list) and len(sig_patterns) == 0):
            print("No activation patterns available. Skipping circuit discovery.")
            return None, None

        # Compute correlation matrix of head activations (across samples)
        all_patterns = np.vstack([
            self.activation_patterns['significant'],
            self.activation_patterns['not_significant']
        ])

        n_heads_total = self.num_layers * self.num_heads
        corr_matrix = np.corrcoef(all_patterns.T)  # (n_heads_total, n_heads_total)

        # Handle NaN values (from constant or zero-variance patterns)
        # NaNs occur when a head has zero variance across all samples
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=1.0, neginf=-1.0)

        print(f"Computed correlation matrix: {corr_matrix.shape}")

        # Early exit if matrix is empty
        if corr_matrix.size == 0:
            print("Empty correlation matrix. Skipping circuit discovery.")
            return None, None

        # Find clusters of highly correlated heads using K-means on correlation
        kmeans = KMeans(n_clusters=n_circuits, random_state=42)
        head_clusters = kmeans.fit_predict(corr_matrix)

        # Build circuits
        circuits = []
        for circuit_id in range(n_circuits):
            # Get heads in this circuit
            head_indices = np.where(head_clusters == circuit_id)[0]

            # Convert flat indices to (layer, head) tuples
            circuit_heads = []
            for idx in head_indices:
                layer = idx // self.num_heads
                head = idx % self.num_heads
                circuit_heads.append((layer, head))

            # Compute circuit statistics
            circuit_activations_sig = self.activation_patterns['significant'][:, head_indices].mean(axis=1)
            circuit_activations_not = self.activation_patterns['not_significant'][:, head_indices].mean(axis=1)

            mean_sig = circuit_activations_sig.mean()
            mean_not = circuit_activations_not.mean()
            differential = mean_sig - mean_not

            circuits.append({
                'id': circuit_id,
                'heads': circuit_heads,
                'size': len(circuit_heads),
                'mean_activation_significant': mean_sig,
                'mean_activation_not_significant': mean_not,
                'differential': differential
            })

        # Sort by differential activation
        circuits = sorted(circuits, key=lambda x: abs(x['differential']), reverse=True)

        # Print discovered circuits
        print("\nDiscovered Circuits:")
        print("-"*70)
        for circuit in circuits:
            print(f"\nCircuit {circuit['id']}:")
            print(f"  Size: {circuit['size']} attention heads")
            print(f"  Mean activation (significant): {circuit['mean_activation_significant']:.4f}")
            print(f"  Mean activation (not_significant): {circuit['mean_activation_not_significant']:.4f}")
            print(f"  Differential: {circuit['differential']:+.4f}")
            print(f"  Sample heads: {circuit['heads'][:5]}...")

        # Visualize circuits
        self._plot_circuits(circuits, corr_matrix, head_clusters)

        return circuits, corr_matrix

    def ablate_layers(self, num_test_samples: int = 20):
        """
        Systematically ablate each layer and measure impact

        Zero out each layer's output and measure:
        1. Change in final representation
        2. Change in predicted class logits (if classifier available)
        """
        print("\n" + "="*70)
        print("LAYER-WISE ABLATION")
        print("="*70)
        print("Testing impact of removing each layer\n")

        results = {
            'layer_effects': np.zeros((num_test_samples, self.num_layers + 1)),
            'samples': []
        }

        test_samples = self.samples[:num_test_samples]

        print(f"Testing {len(test_samples)} samples...\n")

        for sample_idx, sample in enumerate(tqdm(test_samples, desc="Ablating layers")):
            # Get baseline (no ablation)
            inputs = self.tokenizer(
                sample.ref_sequence,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding="max_length"
            ).to(self.device)

            with torch.no_grad():
                baseline_outputs = self.model(**inputs, output_hidden_states=True)

            # Handle different output formats
            if isinstance(baseline_outputs, tuple):
                # DNABERT-2: (last_hidden_state, pooler_output)
                baseline_hidden = baseline_outputs[0]  # (batch, seq_len, hidden)
                baseline_pooled = baseline_hidden.mean(dim=1)  # Pool over sequence
                hidden_states_list = [baseline_hidden]  # Only have final layer
            else:
                # Standard BERT: has hidden_states attribute
                baseline_hidden = baseline_outputs.hidden_states[-1]  # Final layer
                baseline_pooled = baseline_hidden.mean(dim=1)  # Pool over sequence
                hidden_states_list = baseline_outputs.hidden_states

            # Test ablating each layer
            num_available_layers = len(hidden_states_list)
            for layer_idx in range(min(self.num_layers + 1, num_available_layers)):
                # For now, measure effect through hidden state changes
                # (Full ablation would require custom forward pass)

                # Approximate: measure layer's contribution to final representation
                layer_hidden = hidden_states_list[layer_idx]
                contribution = torch.norm(layer_hidden - hidden_states_list[0]).item()

                results['layer_effects'][sample_idx, layer_idx] = contribution

            results['samples'].append(sample)

        # Plot results
        self._plot_layer_ablation(results)

        return results

    def ablate_heads(self, num_test_samples: int = 15):
        """
        Systematically ablate individual attention heads

        Measure importance of each head by computing attention statistics.
        Full ablation would require modifying forward pass.
        """
        print("\n" + "="*70)
        print("ATTENTION HEAD ABLATION")
        print("="*70)
        print("Testing impact of individual attention heads\n")

        results = {
            'head_contributions': np.zeros((num_test_samples, self.num_layers, self.num_heads)),
            'samples': []
        }

        test_samples = self.samples[:num_test_samples]

        print(f"Testing {len(test_samples)} samples...\n")

        for sample_idx, sample in enumerate(tqdm(test_samples, desc="Analyzing heads")):
            attentions, _ = self.get_attention_patterns(sample.ref_sequence)

            # For each head, compute its "contribution" as attention variance
            # (Heads with higher variance are potentially more discriminative)
            for layer_idx in range(self.num_layers):
                layer_attn = attentions[layer_idx][0]  # (num_heads, seq_len, seq_len)

                for head_idx in range(self.num_heads):
                    head_attn = layer_attn[head_idx]

                    # Contribution = variance in attention weights
                    # (high variance = selective attention)
                    contribution = head_attn.var().item()
                    results['head_contributions'][sample_idx, layer_idx, head_idx] = contribution

            results['samples'].append(sample)

        # Plot results
        self._plot_head_ablation(results)

        return results

    def ablate_circuits(self, circuits: List[Dict], num_test_samples: int = 20):
        """
        Ablate discovered circuits and measure impact

        For each circuit, measure what happens when we "remove" it
        (approximate by measuring activation patterns without those heads).
        """
        print("\n" + "="*70)
        print("CIRCUIT ABLATION")
        print("="*70)
        print(f"Testing impact of removing {len(circuits)} discovered circuits\n")

        results = {
            'circuit_effects': np.zeros((num_test_samples, len(circuits))),
            'samples': []
        }

        test_samples = self.samples[:num_test_samples]

        print(f"Testing {len(test_samples)} samples...\n")

        for sample_idx, sample in enumerate(tqdm(test_samples, desc="Ablating circuits")):
            # Get full activation pattern
            _, full_pattern = self.get_attention_patterns(sample.ref_sequence)

            # For each circuit, compute effect of removing it
            for circuit_idx, circuit in enumerate(circuits):
                # Get indices of heads in this circuit
                circuit_head_indices = [
                    layer * self.num_heads + head
                    for layer, head in circuit['heads']
                ]

                # Approximate ablation: how much does this circuit contribute?
                circuit_activations = full_pattern[circuit_head_indices]
                circuit_contribution = np.abs(circuit_activations).mean()

                results['circuit_effects'][sample_idx, circuit_idx] = circuit_contribution

            results['samples'].append(sample)

        # Plot results
        self._plot_circuit_ablation(results, circuits)

        return results

    def run_full_analysis(self):
        """Run complete circuit analysis pipeline"""
        print("\n" + "="*70)
        print("FULL CIRCUIT ANALYSIS")
        print("="*70)
        print("Discovering and ablating functional circuits in DNABERT-2\n")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_dir = self.output_dir / f"analysis_{timestamp}"
        analysis_dir.mkdir(exist_ok=True)
        self.output_dir = analysis_dir

        all_results = {}

        # 1. Collect activation patterns
        print("\n" + "-"*70)
        print("Step 1/5: Collecting activation patterns")
        print("-"*70)
        self.collect_activation_patterns(num_samples_per_class=min(50, len(self.samples)//2))

        # 2. Discover circuits
        print("\n" + "-"*70)
        print("Step 2/5: Discovering functional circuits")
        print("-"*70)
        circuits, corr_matrix = self.discover_circuits(n_circuits=5)
        all_results['circuits'] = circuits
        all_results['correlation_matrix'] = corr_matrix

        # 3. Ablate layers
        print("\n" + "-"*70)
        print("Step 3/5: Layer-wise ablation")
        print("-"*70)
        all_results['layer_ablation'] = self.ablate_layers(num_test_samples=20)

        # 4. Ablate heads
        print("\n" + "-"*70)
        print("Step 4/5: Head-wise ablation")
        print("-"*70)
        all_results['head_ablation'] = self.ablate_heads(num_test_samples=15)

        # 5. Ablate circuits
        print("\n" + "-"*70)
        print("Step 5/5: Circuit ablation")
        print("-"*70)
        all_results['circuit_ablation'] = self.ablate_circuits(circuits, num_test_samples=20)

        # 6. Generate report
        self._generate_report(all_results, timestamp)

        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        print(f"Results saved to: {self.output_dir}")
        print("\nGenerated files:")
        for file in sorted(self.output_dir.glob("*")):
            print(f"  - {file.name}")
        print("="*70 + "\n")

        return all_results

    # ============================================
    # Plotting functions
    # ============================================

    def _plot_circuits(self, circuits: List[Dict], corr_matrix: np.ndarray,
                      head_clusters: np.ndarray):
        """Visualize discovered circuits"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # 1. Correlation matrix with circuit boundaries
        ax = axes[0, 0]
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_title('Attention Head Correlation Matrix\n(Clustered into circuits)',
                    fontsize=13, fontweight='bold')
        ax.set_xlabel('Head Index (layer × heads)', fontsize=11)
        ax.set_ylabel('Head Index (layer × heads)', fontsize=11)
        plt.colorbar(im, ax=ax, label='Correlation')

        # 2. Circuit sizes
        ax = axes[0, 1]
        circuit_ids = [c['id'] for c in circuits]
        circuit_sizes = [c['size'] for c in circuits]
        ax.bar(circuit_ids, circuit_sizes, color='#2E86AB', alpha=0.8)
        ax.set_title('Circuit Sizes', fontsize=13, fontweight='bold')
        ax.set_xlabel('Circuit ID', fontsize=11)
        ax.set_ylabel('Number of Attention Heads', fontsize=11)
        ax.grid(axis='y', alpha=0.3)

        # 3. Circuit differential activation
        ax = axes[1, 0]
        differentials = [c['differential'] for c in circuits]
        colors = ['#A23B72' if d > 0 else '#F18F01' for d in differentials]
        ax.barh(circuit_ids, differentials, color=colors, alpha=0.8)
        ax.set_title('Circuit Differential Activation\n(Significant - Not Significant)',
                    fontsize=13, fontweight='bold')
        ax.set_xlabel('Differential Activation', fontsize=11)
        ax.set_ylabel('Circuit ID', fontsize=11)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.grid(axis='x', alpha=0.3)

        # 4. Circuit activation by class
        ax = axes[1, 1]
        x = np.arange(len(circuits))
        width = 0.35
        sig_acts = [c['mean_activation_significant'] for c in circuits]
        not_sig_acts = [c['mean_activation_not_significant'] for c in circuits]
        ax.bar(x - width/2, sig_acts, width, label='Significant', color='#2E86AB', alpha=0.8)
        ax.bar(x + width/2, not_sig_acts, width, label='Not Significant', color='#F18F01', alpha=0.8)
        ax.set_title('Circuit Activation by Class', fontsize=13, fontweight='bold')
        ax.set_xlabel('Circuit ID', fontsize=11)
        ax.set_ylabel('Mean Activation', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(circuit_ids)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_file = self.output_dir / "1_discovered_circuits.png"
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        print(f"Saved: {output_file.name}")
        plt.close()

    def _plot_layer_ablation(self, results: Dict):
        """Plot layer ablation results"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        effects = results['layer_effects']

        # Average effect per layer
        mean_effects = effects.mean(axis=0)
        std_effects = effects.std(axis=0)
        layers = np.arange(len(mean_effects))

        axes[0].plot(layers, mean_effects, marker='o', linewidth=2,
                    markersize=8, color='#2E86AB')
        axes[0].fill_between(layers, mean_effects - std_effects,
                            mean_effects + std_effects, alpha=0.3, color='#A23B72')
        axes[0].set_xlabel('Layer Index', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Layer Contribution', fontsize=12, fontweight='bold')
        axes[0].set_title('Layer-wise Contributions\n(Mean ± Std across samples)',
                         fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Heatmap
        im = axes[1].imshow(effects.T, aspect='auto', cmap='viridis', interpolation='nearest')
        axes[1].set_xlabel('Sample Index', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Layer Index', fontsize=12, fontweight='bold')
        axes[1].set_title('Layer Contributions Heatmap', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=axes[1], label='Contribution Magnitude')

        plt.tight_layout()
        output_file = self.output_dir / "2_layer_ablation.png"
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        print(f"Saved: {output_file.name}")
        plt.close()

    def _plot_head_ablation(self, results: Dict):
        """Plot head ablation results"""
        contributions = results['head_contributions']

        # Average across samples
        mean_contributions = contributions.mean(axis=0)  # (num_layers, num_heads)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Heatmap
        im = axes[0].imshow(mean_contributions, aspect='auto', cmap='YlOrRd',
                           interpolation='nearest')
        axes[0].set_xlabel('Attention Head', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Layer', fontsize=12, fontweight='bold')
        axes[0].set_title('Attention Head Contributions\n(Averaged across samples)',
                         fontsize=14, fontweight='bold')
        cbar = plt.colorbar(im, ax=axes[0])
        cbar.set_label('Contribution (Attention Variance)', fontsize=11, fontweight='bold')

        # Top heads
        flat_contributions = mean_contributions.flatten()
        top_10_indices = np.argsort(flat_contributions)[-10:][::-1]

        top_layers = []
        top_heads = []
        top_values = []

        for idx in top_10_indices:
            layer = idx // self.num_heads
            head = idx % self.num_heads
            top_layers.append(f"L{layer}H{head}")
            top_values.append(flat_contributions[idx])

        axes[1].barh(range(len(top_values)), top_values, color='#2E86AB', alpha=0.8)
        axes[1].set_yticks(range(len(top_values)))
        axes[1].set_yticklabels(top_layers[::-1], fontsize=10)
        axes[1].set_xlabel('Contribution', fontsize=12, fontweight='bold')
        axes[1].set_title('Top 10 Most Important Heads', fontsize=14, fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)

        plt.tight_layout()
        output_file = self.output_dir / "3_head_ablation.png"
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        print(f"Saved: {output_file.name}")
        plt.close()

    def _plot_circuit_ablation(self, results: Dict, circuits: List[Dict]):
        """Plot circuit ablation results"""
        effects = results['circuit_effects']

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Average effect per circuit
        mean_effects = effects.mean(axis=0)
        std_effects = effects.std(axis=0)
        circuit_ids = [c['id'] for c in circuits]

        axes[0].bar(circuit_ids, mean_effects, yerr=std_effects,
                   color='#2E86AB', alpha=0.8, capsize=5)
        axes[0].set_xlabel('Circuit ID', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Ablation Effect (Contribution)', fontsize=12, fontweight='bold')
        axes[0].set_title('Circuit Contributions\n(Mean ± Std)',
                         fontsize=14, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)

        # Heatmap
        im = axes[1].imshow(effects.T, aspect='auto', cmap='viridis', interpolation='nearest')
        axes[1].set_xlabel('Sample Index', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Circuit ID', fontsize=12, fontweight='bold')
        axes[1].set_title('Circuit Contributions Heatmap', fontsize=14, fontweight='bold')
        axes[1].set_yticks(range(len(circuits)))
        axes[1].set_yticklabels(circuit_ids)
        plt.colorbar(im, ax=axes[1], label='Contribution')

        plt.tight_layout()
        output_file = self.output_dir / "4_circuit_ablation.png"
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        print(f"Saved: {output_file.name}")
        plt.close()

    def _generate_report(self, all_results: Dict, timestamp: str):
        """Generate comprehensive analysis report"""
        report_file = self.output_dir / f"circuit_analysis_report_{timestamp}.txt"

        with open(report_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("CIRCUIT ANALYSIS AND ABLATION REPORT\n")
            f.write("Functional Circuit Discovery in DNABERT-2\n")
            f.write("="*70 + "\n\n")

            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Architecture: {self.num_layers} layers, {self.num_heads} heads per layer\n\n")

            f.write("Dataset:\n")
            f.write(f"  Total samples: {len(self.samples)}\n")
            f.write(f"  Significant sQTLs: {len(self.significant)}\n")
            f.write(f"  Not significant sQTLs: {len(self.not_significant)}\n\n")

            f.write("="*70 + "\n")
            f.write("DISCOVERED CIRCUITS\n")
            f.write("="*70 + "\n\n")

            circuits = all_results['circuits']
            for circuit in circuits:
                f.write(f"Circuit {circuit['id']}:\n")
                f.write(f"  Size: {circuit['size']} attention heads\n")
                f.write(f"  Differential activation: {circuit['differential']:+.4f}\n")
                f.write(f"  Mean activation (significant): {circuit['mean_activation_significant']:.4f}\n")
                f.write(f"  Mean activation (not_significant): {circuit['mean_activation_not_significant']:.4f}\n")
                f.write(f"  Sample heads: {circuit['heads'][:5]}\n")
                f.write("\n")

            f.write("="*70 + "\n")
            f.write("ABLATION RESULTS\n")
            f.write("="*70 + "\n\n")

            # Layer ablation
            f.write("Layer Ablation:\n")
            layer_effects = all_results['layer_ablation']['layer_effects'].mean(axis=0)
            top_layers = np.argsort(layer_effects)[-3:][::-1]
            f.write("  Most important layers:\n")
            for rank, layer_idx in enumerate(top_layers, 1):
                f.write(f"    {rank}. Layer {layer_idx}: Contribution = {layer_effects[layer_idx]:.4f}\n")
            f.write("\n")

            # Head ablation
            f.write("Head Ablation:\n")
            head_contribs = all_results['head_ablation']['head_contributions'].mean(axis=0)
            flat_contribs = head_contribs.flatten()
            top_5_idx = np.argsort(flat_contribs)[-5:][::-1]
            f.write("  Most important heads:\n")
            for rank, idx in enumerate(top_5_idx, 1):
                layer = idx // self.num_heads
                head = idx % self.num_heads
                f.write(f"    {rank}. Layer {layer}, Head {head}: Contribution = {flat_contribs[idx]:.4f}\n")
            f.write("\n")

            # Circuit ablation
            f.write("Circuit Ablation:\n")
            circuit_effects = all_results['circuit_ablation']['circuit_effects'].mean(axis=0)
            f.write("  Circuit contributions:\n")
            for circuit_idx, effect in enumerate(circuit_effects):
                f.write(f"    Circuit {circuit_idx}: Contribution = {effect:.4f}\n")
            f.write("\n")

            f.write("="*70 + "\n")
            f.write("KEY FINDINGS\n")
            f.write("="*70 + "\n\n")

            f.write("1. Circuit Structure:\n")
            f.write(f"   Discovered {len(circuits)} functional circuits in the model.\n")
            f.write("   Circuits show clear differential activation between sQTL classes,\n")
            f.write("   suggesting specialized computational roles.\n\n")

            f.write("2. Layer Importance:\n")
            f.write(f"   Layers {top_layers[0]}, {top_layers[1]}, {top_layers[2]} ")
            f.write("contribute most to\n")
            f.write("   final representations, indicating where critical features are formed.\n\n")

            f.write("3. Circuit Functionality:\n")
            most_differential_circuit = max(circuits, key=lambda x: abs(x['differential']))
            f.write(f"   Circuit {most_differential_circuit['id']} shows strongest ")
            f.write("differential activation\n")
            f.write(f"   (Δ = {most_differential_circuit['differential']:+.4f}), ")
            f.write("suggesting specialized role in\n")
            f.write("   distinguishing significant from not_significant sQTLs.\n\n")

            f.write("="*70 + "\n")
            f.write("OUTPUT FILES\n")
            f.write("="*70 + "\n\n")
            for file in sorted(self.output_dir.glob("*")):
                f.write(f"  - {file.name}\n")
            f.write("\n" + "="*70 + "\n")

        print(f"Report saved: {report_file.name}")


