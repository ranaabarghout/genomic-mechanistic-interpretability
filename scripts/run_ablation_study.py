"""Run ablation study on genomic model."""
import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.load_model import load_base_model, load_finetuned_model
from interpretability.ablation import (
    AblationStudy,
    plot_ablation_results,
    plot_ablation_heatmap,
    find_critical_components
)


def run_ablation_study(
    model_path: str,
    checkpoint_path: str = None,
    sequence: str = None,
    output_dir: str = './ablation_results',
    num_layers: int = 12,
    num_heads: int = 12,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """Run comprehensive ablation study.

    Args:
        model_path: Path to base model
        checkpoint_path: Optional path to fine-tuned checkpoint
        sequence: DNA sequence to test
        output_dir: Directory to save results
        num_layers: Number of layers in model
        num_heads: Number of attention heads per layer
        device: Device to run on
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model from {model_path}...")

    # Load model
    if checkpoint_path:
        model, tokenizer, config = load_finetuned_model(
            checkpoint_path, model_path, device=device
        )
    else:
        model, tokenizer, config = load_base_model(model_path, device=device)

    print(f"Model loaded on {device}")

    # Get model architecture details
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads

    print(f"Model: {num_layers} layers, {num_heads} heads per layer")

    # Prepare input
    if sequence is None:
        sequence = "ATCGATCGATCGATCG" * 8
        print(f"Using example sequence (length: {len(sequence)})")

    inputs = tokenizer(sequence, return_tensors='pt', truncation=True, max_length=512)
    input_ids = inputs['input_ids'].to(device)

    # Define metric function
    def metric_fn(outputs):
        """Extract logits and compute max value as metric."""
        if isinstance(outputs, tuple):
            logits = outputs[0] if hasattr(outputs[0], 'shape') else outputs.logits
        else:
            logits = outputs if hasattr(outputs, 'shape') else outputs.logits
        return logits.max().item()

    # Initialize ablation study
    print("\nInitializing ablation study...")
    ablation = AblationStudy(model, device=device)

    # 1. Scan all layers
    print(f"\n{'='*60}")
    print("LAYER ABLATION STUDY")
    print(f"{'='*60}\n")

    print("Scanning all layers...")
    layer_results = ablation.scan_all_layers(
        input_ids,
        num_layers=num_layers,
        metric_fn=metric_fn,
        ablation_type='zero'
    )

    print(f"\nAblated {len(layer_results)} layers")

    # Find critical layers
    critical_layers = find_critical_components(layer_results, threshold=0.05, top_k=5)
    print(f"\nTop 5 critical layers:")
    for i, result in enumerate(critical_layers, 1):
        print(f"{i}. {result.component_name}: "
              f"importance={result.relative_importance:.4f}, "
              f"diff={result.metric_diff:.4f}")

    # Plot layer results
    layer_plot_path = os.path.join(output_dir, 'layer_ablation.png')
    plot_ablation_results(layer_results, save_path=layer_plot_path, metric_name='Max Logit')

    # 2. Scan attention heads (sample of layers for speed)
    print(f"\n{'='*60}")
    print("ATTENTION HEAD ABLATION STUDY")
    print(f"{'='*60}\n")

    # For large models, only scan a subset of layers
    layers_to_scan = min(6, num_layers)
    print(f"Scanning attention heads in first {layers_to_scan} layers...")

    head_results = {}
    for layer_idx in range(layers_to_scan):
        for head_idx in range(num_heads):
            try:
                result = ablation.zero_ablate_attention_head(
                    input_ids, layer_idx, head_idx, metric_fn
                )
                head_results[result.component_name] = result

                if (head_idx + 1) % 4 == 0:
                    print(f"  Layer {layer_idx}: scanned {head_idx + 1}/{num_heads} heads")
            except Exception as e:
                print(f"Error ablating L{layer_idx}H{head_idx}: {e}")
                continue

    print(f"\nAblated {len(head_results)} attention heads")

    # Find critical heads
    critical_heads = find_critical_components(head_results, threshold=0.05, top_k=10)
    print(f"\nTop 10 critical attention heads:")
    for i, result in enumerate(critical_heads, 1):
        print(f"{i}. {result.component_name}: "
              f"importance={result.relative_importance:.4f}, "
              f"diff={result.metric_diff:.4f}")

    # Plot head results
    head_plot_path = os.path.join(output_dir, 'head_ablation.png')
    plot_ablation_results(head_results, save_path=head_plot_path, metric_name='Max Logit')

    # Plot heatmap
    heatmap_path = os.path.join(output_dir, 'head_ablation_heatmap.png')
    plot_ablation_heatmap(
        head_results,
        num_layers=layers_to_scan,
        num_heads=num_heads,
        save_path=heatmap_path,
        metric='relative_importance'
    )

    # Summary
    print(f"\n{'='*60}")
    print("ABLATION STUDY COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"- Layer ablation: {len(layer_results)} layers")
    print(f"- Head ablation: {len(head_results)} heads")
    print(f"- Critical layers: {len([r for r in layer_results.values() if r.relative_importance >= 0.05])}")
    print(f"- Critical heads: {len([r for r in head_results.values() if r.relative_importance >= 0.05])}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Run ablation study on genomic model')

    parser.add_argument(
        '--model_path',
        type=str,
        default='/project/def-mahadeva/ranaab/genomic-FM/models/dnabert2',
        help='Path to base model'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to fine-tuned checkpoint'
    )

    parser.add_argument(
        '--sequence',
        type=str,
        default=None,
        help='DNA sequence to test'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./ablation_results',
        help='Output directory'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run on'
    )

    args = parser.parse_args()

    run_ablation_study(
        model_path=args.model_path,
        checkpoint_path=args.checkpoint,
        sequence=args.sequence,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == "__main__":
    main()
