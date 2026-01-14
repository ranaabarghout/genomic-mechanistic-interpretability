"""Run comprehensive attention analysis on genomic variants."""
import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.load_model import load_base_model, load_finetuned_model
from interpretability.attention import (
    extract_attention,
    visualize_all_heads,
    compute_attention_rollout,
    analyze_attention_patterns,
    plot_attention_statistics,
    create_interactive_attention_plot
)


def run_attention_analysis(
    model_path: str,
    checkpoint_path: str = None,
    sequence: str = None,
    output_dir: str = './attention_results',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """Run comprehensive attention analysis.

    Args:
        model_path: Path to base model or model name
        checkpoint_path: Optional path to fine-tuned checkpoint
        sequence: DNA sequence to analyze (if None, uses example)
        output_dir: Directory to save outputs
        device: Device to run on
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model from {model_path}...")

    # Load model
    if checkpoint_path:
        print(f"Loading fine-tuned checkpoint from {checkpoint_path}")
        model, tokenizer, config = load_finetuned_model(
            checkpoint_path,
            model_path,
            device=device
        )
    else:
        model, tokenizer, config = load_base_model(
            model_path,
            device=device
        )

    print(f"Model loaded successfully on {device}")
    print(f"Model config: {config}")

    # Use example sequence if none provided
    if sequence is None:
        # Example: sequence with a known variant
        sequence = "ATCGATCGATCGATCG" * 8  # 128bp sequence
        print(f"Using example sequence (length: {len(sequence)})")
    else:
        print(f"Analyzing sequence (length: {len(sequence)})")

    # Tokenize
    print("\nTokenizing sequence...")
    inputs = tokenizer(
        sequence,
        return_tensors='pt',
        truncation=True,
        max_length=512
    ).to(device)

    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    print(f"Number of tokens: {len(tokens)}")

    # Extract attention
    print("\nExtracting attention patterns...")
    num_layers = config.num_hidden_layers

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        attentions = outputs.attentions  # Tuple of [batch, heads, seq, seq]

    print(f"Extracted attention from {len(attentions)} layers")

    # Analyze each layer
    print("\nAnalyzing attention patterns...")

    for layer_idx in range(min(3, num_layers)):  # Analyze first 3 layers
        print(f"\n=== Layer {layer_idx} ===")

        attention = attentions[layer_idx]  # [batch, heads, seq, seq]

        # Visualize all heads
        print(f"Visualizing all attention heads...")
        vis_path = os.path.join(output_dir, f'layer{layer_idx}_all_heads.png')
        visualize_all_heads(attention, tokens, layer_idx, save_path=vis_path)

        # Compute basic statistics for this layer
        print(f"Computing attention statistics...")
        attn_np = attention[0].cpu().numpy()  # [heads, seq_len, seq_len]
        print(f"Mean attention: {attn_np.mean():.4f}")
        print(f"Std attention: {attn_np.std():.4f}")
        print(f"Max attention: {attn_np.max():.4f}")
        print(f"Min attention: {attn_np.min():.4f}")

    # Analyze patterns across all layers
    print("\n=== Cross-Layer Analysis ===")
    print("Computing attention rollout and token importance...")
    results = analyze_attention_patterns(attentions, tokens)

    print(f"\nMost attended tokens across all layers:")
    for token, score in results['most_attended_tokens'][:5]:
        print(f"  {token}: {score:.4f}")

    # Create interactive plot for first layer
    print(f"\nCreating interactive visualization for layer 0...")
    interactive_path = os.path.join(output_dir, 'layer0_interactive.html')
    fig = create_interactive_attention_plot(attentions[0], tokens, title="Layer 0 Attention")
    fig.write_html(interactive_path)
    print(f"Interactive plot saved to {interactive_path}")

    # Visualize attention rollout
    print("\n=== Visualizing Attention Rollout ===")
    rollout = results['rollout']

    # Visualize rollout
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(rollout.cpu().numpy(), cmap='viridis', aspect='auto')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_title('Attention Rollout (All Layers)')
    plt.colorbar(im, ax=ax, label='Attention Weight')

    rollout_path = os.path.join(output_dir, 'attention_rollout.png')
    plt.savefig(rollout_path, dpi=300, bbox_inches='tight')
    print(f"Rollout visualization saved to {rollout_path}")
    plt.close()

    # Summary
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"- Layer visualizations: {num_layers} layers")
    print(f"- Attention statistics: {min(3, num_layers)} layers")
    print(f"- Interactive plot: layer 0")
    print(f"- Attention rollout: all layers")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Run attention analysis on genomic sequences')

    parser.add_argument(
        '--model_path',
        type=str,
        default='/project/def-mahadeva/ranaab/genomic-FM/models/dnabert2',
        help='Path to base model or HuggingFace model name'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to fine-tuned checkpoint (.ckpt file)'
    )

    parser.add_argument(
        '--sequence',
        type=str,
        default=None,
        help='DNA sequence to analyze (ATCG format)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./attention_results',
        help='Directory to save output plots'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run on (cuda/cpu)'
    )

    args = parser.parse_args()

    run_attention_analysis(
        model_path=args.model_path,
        checkpoint_path=args.checkpoint,
        sequence=args.sequence,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == "__main__":
    main()

