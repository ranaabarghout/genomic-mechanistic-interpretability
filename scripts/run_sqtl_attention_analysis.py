"""
sQTL Attention Analysis with Original Sequences
==============================================
Analyzes attention patterns in DNABERT-2 model on original sQTL DNA sequences.

This script:
1. Loads original DNA sequences (not PCA-transformed)
2. Tokenizes sequences using DNABERT-2 tokenizer
3. Computes attention weights across all layers
4. Visualizes which nucleotides the model focuses on
5. Compares attention between significant vs non-significant sQTLs
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
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import data loader and model loader
from data.sqtl_data_loader import OriginalSQTLDataLoader
from models.load_model import load_base_model, load_finetuned_model

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


class sQTLAttentionAnalyzer:
    """Analyze attention patterns on original sQTL sequences"""

    def __init__(self,
                 checkpoint_path: str = None,
                 model_name: str = "zhihan1996/DNABERT-2-117M",
                 num_samples: int = 50,
                 output_dir: str = "outputs/sqtl_attention_original"):
        """
        Initialize analyzer

        Args:
            checkpoint_path: Path to fine-tuned checkpoint (optional)
            model_name: Base model name
            num_samples: Number of samples to analyze
            output_dir: Output directory for results
        """
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.model_name = model_name
        self.num_samples = num_samples
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load model and tokenizer
        print("\nLoading model...")
        if self.checkpoint_path and self.checkpoint_path.exists():
            self.model, self.tokenizer, self.config = load_finetuned_model(
                str(self.checkpoint_path),
                model_name,
                device=str(self.device)
            )
        else:
            self.model, self.tokenizer, self.config = load_base_model(
                model_name,
                device=str(self.device)
            )

        print(f"\nModel configuration:")
        print(f"  Layers: {self.config.num_hidden_layers}")
        print(f"  Attention heads: {self.config.num_attention_heads}")
        print(f"  Hidden size: {self.config.hidden_size}")

        # Load data
        print("\nLoading sQTL data...")
        loader = OriginalSQTLDataLoader(num_samples=num_samples, tissue_filter="Whole_Blood")
        self.samples = loader.load_data()
        loader.print_statistics(self.samples)

    def compute_attention(self, sequence: str, max_length: int = 512):
        """
        Compute attention weights for a DNA sequence

        Args:
            sequence: DNA sequence string
            max_length: Maximum sequence length for model

        Returns:
            Dictionary with attention weights and tokens
        """
        # Tokenize sequence
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding="max_length"
        ).to(self.device)

        # Get model outputs with attention
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        # Extract attention weights
        # attentions is a tuple of (num_layers, batch_size, num_heads, seq_len, seq_len)
        attentions = outputs.attentions

        # Convert tokens to readable format
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        # Stack attention weights (layers, heads, seq_len, seq_len)
        attention_weights = torch.stack([attn[0] for attn in attentions])  # Remove batch dim

        return {
            'attention': attention_weights.cpu().numpy(),
            'tokens': tokens,
            'input_ids': inputs['input_ids'][0].cpu().numpy(),
            'sequence': sequence[:max_length]
        }

    def analyze_attention_by_class(self, num_samples_per_class: int = 10):
        """Analyze average attention patterns for each class"""
        print("\n" + "="*60)
        print("Analyzing Attention by Class")
        print("="*60)

        # Separate samples by class
        significant = [s for s in self.samples if s.label == 0][:num_samples_per_class]
        not_significant = [s for s in self.samples if s.label == 1][:num_samples_per_class]

        print(f"Analyzing {len(significant)} significant and {len(not_significant)} not_significant samples")

        # Compute average attention for each class
        results = {}
        for class_name, samples in [('significant', significant), ('not_significant', not_significant)]:
            print(f"\nProcessing {class_name} samples...")

            all_attentions = []
            for sample in tqdm(samples, desc=f"{class_name}"):
                result = self.compute_attention(sample.ref_sequence)
                # Average across heads: (layers, seq_len, seq_len)
                attn = result['attention'].mean(axis=1)
                all_attentions.append(attn)

            # Average across samples: (layers, seq_len, seq_len)
            avg_attention = np.mean(all_attentions, axis=0)

            results[class_name] = {
                'attention': avg_attention,
                'tokens': result['tokens'],  # Same tokenization
                'num_samples': len(samples)
            }

        # Plot comparison
        self._plot_attention_comparison(results)

        return results

    def analyze_variant_position_attention(self, num_samples: int = 20):
        """
        Analyze attention focused on variant positions

        For each sample, compute how much attention is focused on the
        variant position (where ref and alt sequences differ).
        """
        print("\n" + "="*60)
        print("Analyzing Attention at Variant Positions")
        print("="*60)

        variant_attentions = []

        for sample in tqdm(self.samples[:num_samples], desc="Analyzing"):
            if sample.variant_pos < 0:
                continue

            # Compute attention for reference sequence
            result = self.compute_attention(sample.ref_sequence)
            attention = result['attention']  # (layers, heads, seq_len, seq_len)

            # Get token position corresponding to variant
            # DNABERT-2 uses k-mer tokenization, so need to map carefully
            # For simplicity, use approximate mapping
            seq_len = attention.shape[2]
            token_pos = min(int(sample.variant_pos * seq_len / 1024), seq_len - 1)

            # Average attention TO the variant position across all positions
            # Shape: (layers, heads, seq_len)
            attn_to_variant = attention[:, :, :, token_pos]

            # Average across heads and source positions
            avg_attn = attn_to_variant.mean(axis=(1, 2))  # (layers,)

            variant_attentions.append({
                'label': sample.label_name,
                'variant_pos': sample.variant_pos,
                'token_pos': token_pos,
                'attention': avg_attn
            })

        # Plot results
        self._plot_variant_attention(variant_attentions)

        return variant_attentions

    def visualize_sample_attention(self, sample_idx: int = 0, layer: int = 11):
        """
        Create detailed attention visualization for a single sample

        Args:
            sample_idx: Index of sample to visualize
            layer: Which layer to visualize (default: last layer)
        """
        sample = self.samples[sample_idx]
        print(f"\n" + "="*60)
        print(f"Visualizing Sample {sample_idx}")
        print(f"  {sample}")
        print("="*60)

        # Compute attention for both ref and alt sequences
        ref_result = self.compute_attention(sample.ref_sequence)
        alt_result = self.compute_attention(sample.alt_sequence)

        # Extract specific layer attention (average across heads)
        ref_attn = ref_result['attention'][layer].mean(axis=0)  # (seq_len, seq_len)
        alt_attn = alt_result['attention'][layer].mean(axis=0)

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        # Plot reference attention
        im1 = axes[0].imshow(ref_attn, cmap='viridis', aspect='auto')
        axes[0].set_title(f'Reference Sequence\n(Label: {sample.label_name})', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Token Position')
        axes[0].set_ylabel('Token Position')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        # Plot alternate attention
        im2 = axes[1].imshow(alt_attn, cmap='viridis', aspect='auto')
        axes[1].set_title(f'Alternate Sequence\n(Variant at position {sample.variant_pos})', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Token Position')
        axes[1].set_ylabel('Token Position')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        # Plot difference
        diff = alt_attn - ref_attn
        im3 = axes[2].imshow(diff, cmap='RdBu_r', aspect='auto', vmin=-diff.std(), vmax=diff.std())
        axes[2].set_title('Attention Difference\n(Alt - Ref)', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Token Position')
        axes[2].set_ylabel('Token Position')
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()

        # Save figure
        output_file = self.output_dir / f"sample_{sample_idx}_layer{layer}_attention.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {output_file}")
        plt.close()

        return ref_result, alt_result

    def _plot_attention_comparison(self, results: dict):
        """Plot comparison of attention patterns between classes"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        for idx, (class_name, data) in enumerate(results.items()):
            row = idx

            # Get last layer attention
            last_layer_attn = data['attention'][-1]  # (seq_len, seq_len)

            # Plot full attention matrix
            im = axes[row, 0].imshow(last_layer_attn, cmap='viridis', aspect='auto')
            axes[row, 0].set_title(f'{class_name.replace("_", " ").title()}\nFull Attention Matrix (Layer 11)',
                                   fontsize=12, fontweight='bold')
            axes[row, 0].set_xlabel('Target Position')
            axes[row, 0].set_ylabel('Source Position')
            plt.colorbar(im, ax=axes[row, 0], fraction=0.046, pad=0.04)

            # Plot attention distribution (average attention per position)
            avg_attn_per_pos = last_layer_attn.mean(axis=0)
            axes[row, 1].plot(avg_attn_per_pos, linewidth=2)
            axes[row, 1].set_title(f'{class_name.replace("_", " ").title()}\nAverage Attention per Position',
                                   fontsize=12, fontweight='bold')
            axes[row, 1].set_xlabel('Token Position')
            axes[row, 1].set_ylabel('Average Attention')
            axes[row, 1].grid(True, alpha=0.3)

            # Mark variant position (typically around 512 in sequence, maps to ~256 in tokens)
            axes[row, 1].axvline(x=len(avg_attn_per_pos)//2, color='red',
                                linestyle='--', alpha=0.5, label='Approx. Variant Position')
            axes[row, 1].legend()

        plt.tight_layout()

        output_file = self.output_dir / "attention_by_class_comparison.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {output_file}")
        plt.close()

    def _plot_variant_attention(self, variant_attentions: list):
        """Plot attention focused on variant positions"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Separate by class
        significant = [v for v in variant_attentions if v['label'] == 'positive']
        not_significant = [v for v in variant_attentions if v['label'] == 'negative']

        # Plot layer-wise attention for each class
        for data_list, label, color in [(significant, 'Significant', 'blue'),
                                         (not_significant, 'Not Significant', 'orange')]:
            if not data_list:
                continue

            # Stack attention across samples: (num_samples, num_layers)
            attns = np.array([v['attention'] for v in data_list])

            # Plot mean and std
            mean_attn = attns.mean(axis=0)
            std_attn = attns.std(axis=0)
            layers = np.arange(len(mean_attn))

            axes[0].plot(layers, mean_attn, label=label, marker='o', linewidth=2, color=color)
            axes[0].fill_between(layers, mean_attn - std_attn, mean_attn + std_attn,
                                alpha=0.2, color=color)

        axes[0].set_title('Attention to Variant Position\nAcross Layers',
                         fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Layer')
        axes[0].set_ylabel('Average Attention')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot distribution of variant positions
        all_positions = [v['variant_pos'] for v in variant_attentions]
        axes[1].hist(all_positions, bins=30, alpha=0.7, edgecolor='black')
        axes[1].set_title('Distribution of Variant Positions', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Position in Sequence (bp)')
        axes[1].set_ylabel('Frequency')
        axes[1].axvline(x=512, color='red', linestyle='--', label='Expected Center (512bp)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        output_file = self.output_dir / "variant_position_attention.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {output_file}")
        plt.close()

    def run_full_analysis(self):
        """Run complete attention analysis"""
        print("\n" + "="*60)
        print("Running Full Attention Analysis")
        print("="*60)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Attention by class
        print("\n1. Analyzing attention patterns by class...")
        class_results = self.analyze_attention_by_class(num_samples_per_class=10)

        # 2. Variant position attention
        print("\n2. Analyzing attention at variant positions...")
        variant_results = self.analyze_variant_position_attention(num_samples=20)

        # 3. Visualize specific examples
        print("\n3. Creating detailed visualizations for example samples...")
        for idx in [0, 1, 2]:
            if idx < len(self.samples):
                self.visualize_sample_attention(sample_idx=idx, layer=11)

        # 4. Generate summary report
        self._generate_report(class_results, variant_results, timestamp)

        print("\n" + "="*60)
        print("Analysis Complete!")
        print(f"Results saved to: {self.output_dir}")
        print("="*60)

    def _generate_report(self, class_results: dict, variant_results: list, timestamp: str):
        """Generate summary report"""
        report_file = self.output_dir / f"attention_analysis_report_{timestamp}.txt"

        with open(report_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("sQTL Attention Analysis Report\n")
            f.write("="*60 + "\n\n")

            f.write(f"Model: {self.model_name}\n")
            if self.checkpoint_path:
                f.write(f"Checkpoint: {self.checkpoint_path}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Device: {self.device}\n\n")

            f.write("Dataset:\n")
            f.write(f"  Total samples analyzed: {len(self.samples)}\n")
            f.write(f"  Significant: {sum(s.label == 0 for s in self.samples)}\n")
            f.write(f"  Not significant: {sum(s.label == 1 for s in self.samples)}\n\n")

            f.write("Model Configuration:\n")
            f.write(f"  Layers: {self.config.num_hidden_layers}\n")
            f.write(f"  Attention heads: {self.config.num_attention_heads}\n")
            f.write(f"  Hidden size: {self.config.hidden_size}\n\n")

            f.write("Analyses Performed:\n")
            f.write("  1. Attention patterns by class (significant vs not_significant)\n")
            f.write("  2. Attention focused on variant positions\n")
            f.write("  3. Detailed sample-level attention visualizations\n\n")

            f.write("Key Findings:\n")
            f.write(f"  - Analyzed {len(variant_results)} samples for variant attention\n")
            f.write(f"  - Average variant position: {np.mean([v['variant_pos'] for v in variant_results]):.1f}bp\n")
            f.write(f"  - Median variant position: {np.median([v['variant_pos'] for v in variant_results]):.1f}bp\n\n")

            f.write("Output Files:\n")
            f.write(f"  - attention_by_class_comparison.png\n")
            f.write(f"  - variant_position_attention.png\n")
            f.write(f"  - sample_*_layer*_attention.png (individual samples)\n\n")

            f.write("="*60 + "\n")

        print(f"\nReport saved: {report_file}")


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description="sQTL Attention Analysis")
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to fine-tuned checkpoint')
    parser.add_argument('--num-samples', type=int, default=50,
                       help='Number of samples to analyze')
    parser.add_argument('--output-dir', type=str, default='outputs/sqtl_attention_original',
                       help='Output directory')

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = sQTLAttentionAnalyzer(
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )

    # Run analysis
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
