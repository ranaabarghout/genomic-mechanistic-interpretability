"""Attention visualization and analysis utilities."""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def extract_attention(model, inputs, layer_idx: Optional[int] = None):
    """Extract attention weights from model.

    Args:
        model: Transformer model with attention
        inputs: Tokenized inputs
        layer_idx: Optional specific layer index

    Returns:
        Attention tensors (list of tensors per layer or single tensor)
    """
    with torch.no_grad():
        outputs = model(**inputs)
        attentions = outputs.attentions if hasattr(outputs, 'attentions') else outputs[-1]

    if layer_idx is not None:
        return attentions[layer_idx]
    return attentions


def visualize_attention_head(
    attention: torch.Tensor,
    tokens: List[str],
    head_idx: int = 0,
    title: str = "Attention Pattern",
    save_path: Optional[str] = None
):
    """Visualize a single attention head.

    Args:
        attention: Attention tensor [batch, heads, seq_len, seq_len]
        tokens: List of token strings
        head_idx: Which attention head to visualize
        title: Plot title
        save_path: Optional path to save figure
    """
    # Extract single head
    attn_head = attention[0, head_idx].cpu().numpy()

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        attn_head,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='viridis',
        cbar_kws={'label': 'Attention Weight'},
        square=True,
        ax=ax
    )

    ax.set_title(f'{title} - Head {head_idx}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Key Tokens', fontsize=12)
    ax.set_ylabel('Query Tokens', fontsize=12)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention plot saved to {save_path}")

    plt.show()
    return fig


def visualize_all_heads(
    attention: torch.Tensor,
    tokens: List[str],
    layer_idx: int = 0,
    save_path: Optional[str] = None
):
    """Visualize all attention heads in a layer.

    Args:
        attention: Attention tensor [batch, heads, seq_len, seq_len]
        tokens: List of token strings
        layer_idx: Layer index
        save_path: Optional path to save figure
    """
    num_heads = attention.shape[1]
    cols = 4
    rows = (num_heads + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
    axes = axes.flatten() if num_heads > 1 else [axes]

    for head_idx in range(num_heads):
        ax = axes[head_idx]
        attn_head = attention[0, head_idx].cpu().numpy()

        sns.heatmap(
            attn_head,
            xticklabels=tokens if len(tokens) <= 20 else False,
            yticklabels=tokens if len(tokens) <= 20 else False,
            cmap='viridis',
            cbar=True,
            square=True,
            ax=ax
        )

        ax.set_title(f'Head {head_idx}', fontsize=10)

        if len(tokens) <= 20:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    # Hide empty subplots
    for idx in range(num_heads, len(axes)):
        axes[idx].axis('off')

    fig.suptitle(f'All Attention Heads - Layer {layer_idx}', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Multi-head attention plot saved to {save_path}")

    plt.show()
    return fig


def compute_attention_rollout(attentions: List[torch.Tensor], discard_ratio: float = 0.1):
    """Compute attention rollout across all layers.

    Attention rollout accumulates attention weights across layers to show
    which input tokens most influence each output token.

    Args:
        attentions: List of attention tensors from each layer
        discard_ratio: Ratio of lowest attention weights to discard

    Returns:
        Rolled out attention matrix
    """
    # Initialize result on the same device as the attention tensors
    device = attentions[0].device
    result = torch.eye(attentions[0].size(-1), device=device)

    with torch.no_grad():
        for attention in attentions:
            # Average attention across all heads
            attention_heads_fused = attention.mean(dim=1)[0]

            # Apply discard ratio
            flat = attention_heads_fused.view(-1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), largest=False)
            flat[indices] = 0

            # Normalize
            attention_heads_fused = attention_heads_fused / attention_heads_fused.sum(dim=-1, keepdim=True)

            # Accumulate
            result = torch.matmul(attention_heads_fused, result)

    return result


def analyze_attention_patterns(
    attentions: List[torch.Tensor],
    tokens: List[str]
) -> Dict:
    """Analyze attention patterns across layers.

    Args:
        attentions: List of attention tensors from each layer
        tokens: List of token strings

    Returns:
        Dictionary with analysis results
    """
    results = {}

    # Compute attention statistics per layer
    layer_stats = []
    for layer_idx, attn in enumerate(attentions):
        attn_np = attn[0].cpu().numpy()  # [heads, seq_len, seq_len]

        stats = {
            'layer': layer_idx,
            'mean': attn_np.mean(),
            'std': attn_np.std(),
            'max': attn_np.max(),
            'min': attn_np.min(),
            'entropy': compute_attention_entropy(attn)
        }
        layer_stats.append(stats)

    results['layer_stats'] = layer_stats

    # Compute attention rollout
    results['rollout'] = compute_attention_rollout(attentions)

    # Find most attended tokens
    avg_attention = torch.stack([a[0].mean(dim=0).mean(dim=0) for a in attentions]).mean(dim=0)
    top_k = 10
    top_indices = avg_attention.topk(min(top_k, len(tokens))).indices.cpu().numpy()
    results['most_attended_tokens'] = [(tokens[i], avg_attention[i].item()) for i in top_indices]

    return results


def compute_attention_entropy(attention: torch.Tensor) -> float:
    """Compute entropy of attention distribution.

    Higher entropy indicates more diffuse attention.

    Args:
        attention: Attention tensor

    Returns:
        Mean entropy across all attention distributions
    """
    # Flatten to [batch * heads * queries, keys]
    attn_flat = attention.view(-1, attention.size(-1))

    # Compute entropy
    entropy = -(attn_flat * torch.log(attn_flat + 1e-9)).sum(dim=-1)

    return entropy.mean().item()


def plot_attention_statistics(results: Dict, save_path: Optional[str] = None):
    """Plot attention statistics across layers.

    Args:
        results: Results from analyze_attention_patterns
        save_path: Optional path to save figure
    """
    layer_stats = results['layer_stats']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Attention Statistics Across Layers', fontsize=16, fontweight='bold')

    layers = [s['layer'] for s in layer_stats]

    # Mean attention
    ax = axes[0, 0]
    means = [s['mean'] for s in layer_stats]
    ax.plot(layers, means, marker='o', linewidth=2)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean Attention')
    ax.set_title('Mean Attention per Layer')
    ax.grid(True, alpha=0.3)

    # Attention entropy
    ax = axes[0, 1]
    entropies = [s['entropy'] for s in layer_stats]
    ax.plot(layers, entropies, marker='s', linewidth=2, color='orange')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Entropy')
    ax.set_title('Attention Entropy per Layer')
    ax.grid(True, alpha=0.3)

    # Attention range (max - min)
    ax = axes[1, 0]
    ranges = [s['max'] - s['min'] for s in layer_stats]
    ax.plot(layers, ranges, marker='^', linewidth=2, color='green')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Range (Max - Min)')
    ax.set_title('Attention Range per Layer')
    ax.grid(True, alpha=0.3)

    # Most attended tokens
    ax = axes[1, 1]
    tokens, weights = zip(*results['most_attended_tokens'])
    y_pos = np.arange(len(tokens))
    ax.barh(y_pos, weights, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tokens)
    ax.set_xlabel('Average Attention Weight')
    ax.set_title('Most Attended Tokens')
    ax.invert_yaxis()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Statistics plot saved to {save_path}")

    plt.show()
    return fig


def create_interactive_attention_plot(
    attention: torch.Tensor,
    tokens: List[str],
    title: str = "Interactive Attention Visualization"
) -> go.Figure:
    """Create interactive attention visualization with Plotly.

    Args:
        attention: Attention tensor [batch, heads, seq_len, seq_len]
        tokens: List of token strings
        title: Plot title

    Returns:
        Plotly figure object
    """
    num_heads = attention.shape[1]

    # Create subplots
    fig = make_subplots(
        rows=(num_heads + 3) // 4,
        cols=4,
        subplot_titles=[f'Head {i}' for i in range(num_heads)],
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )

    for head_idx in range(num_heads):
        row = head_idx // 4 + 1
        col = head_idx % 4 + 1

        attn_head = attention[0, head_idx].cpu().numpy()

        heatmap = go.Heatmap(
            z=attn_head,
            x=tokens,
            y=tokens,
            colorscale='Viridis',
            showscale=(head_idx == 0)
        )

        fig.add_trace(heatmap, row=row, col=col)

    fig.update_layout(
        title_text=title,
        height=300 * ((num_heads + 3) // 4),
        showlegend=False
    )

    return fig

