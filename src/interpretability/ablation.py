"""Ablation studies for understanding component importance in neural networks."""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from tqdm import tqdm
import copy


@dataclass
class AblationResult:
    """Results from an ablation experiment."""
    component_name: str
    component_type: str  # 'head', 'layer', 'neuron'
    original_metric: float
    ablated_metric: float
    metric_diff: float
    relative_importance: float


class AblationStudy:
    """Performs systematic ablation studies on neural networks."""
    
    def __init__(
        self,
        model,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """Initialize ablation study.
        
        Args:
            model: PyTorch model to analyze
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def zero_ablate_attention_head(
        self,
        inputs: torch.Tensor,
        layer_idx: int,
        head_idx: int,
        metric_fn: Callable
    ) -> AblationResult:
        """Ablate a specific attention head by zeroing its output.
        
        Args:
            inputs: Input tensor
            layer_idx: Index of transformer layer
            head_idx: Index of attention head
            metric_fn: Function to compute performance metric
            
        Returns:
            AblationResult with ablation impact
        """
        inputs = inputs.to(self.device)
        
        # Get original output
        with torch.no_grad():
            original_output = self.model(inputs)
            original_metric = metric_fn(original_output)
        
        # Create hook to zero out attention head
        def ablation_hook(module, input, output):
            # output is typically (hidden_states, attention_weights)
            if isinstance(output, tuple):
                hidden_states = output[0]
                # Assuming multi-head attention: [batch, seq, num_heads, head_dim]
                # Zero out the specific head
                if len(hidden_states.shape) == 4:
                    hidden_states[:, :, head_idx, :] = 0
                return (hidden_states,) + output[1:]
            return output
        
        # Find target layer
        target_layer = None
        for name, module in self.model.named_modules():
            if f'layer.{layer_idx}' in name and 'attention' in name.lower():
                target_layer = module
                break
        
        if target_layer is None:
            raise ValueError(f"Could not find attention layer {layer_idx}")
        
        # Register hook and run ablated model
        handle = target_layer.register_forward_hook(ablation_hook)
        
        with torch.no_grad():
            ablated_output = self.model(inputs)
            ablated_metric = metric_fn(ablated_output)
        
        handle.remove()
        
        # Compute impact
        metric_diff = ablated_metric - original_metric
        relative_importance = abs(metric_diff) / (abs(original_metric) + 1e-10)
        
        return AblationResult(
            component_name=f'L{layer_idx}H{head_idx}',
            component_type='head',
            original_metric=original_metric,
            ablated_metric=ablated_metric,
            metric_diff=metric_diff,
            relative_importance=relative_importance
        )
    
    def mean_ablate_attention_head(
        self,
        inputs: torch.Tensor,
        layer_idx: int,
        head_idx: int,
        metric_fn: Callable,
        reference_inputs: Optional[torch.Tensor] = None
    ) -> AblationResult:
        """Ablate attention head by replacing with mean activation.
        
        Args:
            inputs: Input tensor
            layer_idx: Layer index
            head_idx: Head index
            metric_fn: Metric function
            reference_inputs: Optional reference inputs to compute mean
            
        Returns:
            AblationResult
        """
        # If no reference provided, use current input
        if reference_inputs is None:
            reference_inputs = inputs
        
        # Compute mean activation for this head
        activations = []
        
        def capture_hook(module, input, output):
            if isinstance(output, tuple):
                activations.append(output[0].detach())
        
        # Find target layer
        target_layer = None
        for name, module in self.model.named_modules():
            if f'layer.{layer_idx}' in name and 'attention' in name.lower():
                target_layer = module
                break
        
        if target_layer is None:
            raise ValueError(f"Could not find attention layer {layer_idx}")
        
        # Capture activations from reference
        handle = target_layer.register_forward_hook(capture_hook)
        
        with torch.no_grad():
            _ = self.model(reference_inputs.to(self.device))
        
        handle.remove()
        
        # Compute mean for specific head
        if len(activations) > 0 and len(activations[0].shape) == 4:
            mean_activation = activations[0][:, :, head_idx, :].mean(dim=(0, 1), keepdim=True)
        else:
            mean_activation = None
        
        # Get original metric
        with torch.no_grad():
            original_output = self.model(inputs)
            original_metric = metric_fn(original_output)
        
        # Create ablation hook
        def ablation_hook(module, input, output):
            if isinstance(output, tuple) and mean_activation is not None:
                hidden_states = output[0]
                if len(hidden_states.shape) == 4:
                    hidden_states[:, :, head_idx, :] = mean_activation
                return (hidden_states,) + output[1:]
            return output
        
        # Run with ablation
        handle = target_layer.register_forward_hook(ablation_hook)
        
        with torch.no_grad():
            ablated_output = self.model(inputs)
            ablated_metric = metric_fn(ablated_output)
        
        handle.remove()
        
        metric_diff = ablated_metric - original_metric
        relative_importance = abs(metric_diff) / (abs(original_metric) + 1e-10)
        
        return AblationResult(
            component_name=f'L{layer_idx}H{head_idx}',
            component_type='head',
            original_metric=original_metric,
            ablated_metric=ablated_metric,
            metric_diff=metric_diff,
            relative_importance=relative_importance
        )
    
    def ablate_layer(
        self,
        inputs: torch.Tensor,
        layer_idx: int,
        metric_fn: Callable,
        ablation_type: str = 'zero'
    ) -> AblationResult:
        """Ablate an entire layer.
        
        Args:
            inputs: Input tensor
            layer_idx: Index of layer to ablate
            metric_fn: Metric function
            ablation_type: 'zero' or 'identity' (skip connection)
            
        Returns:
            AblationResult
        """
        inputs = inputs.to(self.device)
        
        # Get original output
        with torch.no_grad():
            original_output = self.model(inputs)
            original_metric = metric_fn(original_output)
        
        # Find target layer
        target_layer = None
        for name, module in self.model.named_modules():
            if f'layer.{layer_idx}' in name or f'layers.{layer_idx}' in name:
                if 'LayerNorm' not in name:  # Skip normalization layers
                    target_layer = module
                    break
        
        if target_layer is None:
            raise ValueError(f"Could not find layer {layer_idx}")
        
        # Create ablation hook
        def ablation_hook(module, input, output):
            if ablation_type == 'zero':
                # Zero out all outputs
                if isinstance(output, tuple):
                    return tuple(torch.zeros_like(o) for o in output)
                else:
                    return torch.zeros_like(output)
            elif ablation_type == 'identity':
                # Pass through input (skip connection)
                if isinstance(output, tuple):
                    return input[0] if len(input) > 0 else output
                else:
                    return input[0] if len(input) > 0 else output
            return output
        
        # Run with ablation
        handle = target_layer.register_forward_hook(ablation_hook)
        
        with torch.no_grad():
            ablated_output = self.model(inputs)
            ablated_metric = metric_fn(ablated_output)
        
        handle.remove()
        
        metric_diff = ablated_metric - original_metric
        relative_importance = abs(metric_diff) / (abs(original_metric) + 1e-10)
        
        return AblationResult(
            component_name=f'Layer{layer_idx}',
            component_type='layer',
            original_metric=original_metric,
            ablated_metric=ablated_metric,
            metric_diff=metric_diff,
            relative_importance=relative_importance
        )
    
    def scan_all_heads(
        self,
        inputs: torch.Tensor,
        num_layers: int,
        num_heads: int,
        metric_fn: Callable,
        ablation_method: str = 'zero'
    ) -> Dict[str, AblationResult]:
        """Systematically ablate all attention heads.
        
        Args:
            inputs: Input tensor
            num_layers: Number of layers in model
            num_heads: Number of attention heads per layer
            metric_fn: Metric function
            ablation_method: 'zero' or 'mean'
            
        Returns:
            Dictionary of ablation results
        """
        results = {}
        
        for layer_idx in tqdm(range(num_layers), desc="Scanning layers"):
            for head_idx in range(num_heads):
                try:
                    if ablation_method == 'zero':
                        result = self.zero_ablate_attention_head(
                            inputs, layer_idx, head_idx, metric_fn
                        )
                    elif ablation_method == 'mean':
                        result = self.mean_ablate_attention_head(
                            inputs, layer_idx, head_idx, metric_fn
                        )
                    else:
                        raise ValueError(f"Unknown ablation method: {ablation_method}")
                    
                    results[result.component_name] = result
                except Exception as e:
                    print(f"Error ablating L{layer_idx}H{head_idx}: {e}")
                    continue
        
        return results
    
    def scan_all_layers(
        self,
        inputs: torch.Tensor,
        num_layers: int,
        metric_fn: Callable,
        ablation_type: str = 'zero'
    ) -> Dict[str, AblationResult]:
        """Systematically ablate all layers.
        
        Args:
            inputs: Input tensor
            num_layers: Number of layers
            metric_fn: Metric function
            ablation_type: 'zero' or 'identity'
            
        Returns:
            Dictionary of ablation results
        """
        results = {}
        
        for layer_idx in tqdm(range(num_layers), desc="Scanning layers"):
            try:
                result = self.ablate_layer(
                    inputs, layer_idx, metric_fn, ablation_type
                )
                results[result.component_name] = result
            except Exception as e:
                print(f"Error ablating layer {layer_idx}: {e}")
                continue
        
        return results


def plot_ablation_results(
    results: Dict[str, AblationResult],
    save_path: Optional[str] = None,
    metric_name: str = 'Metric'
):
    """Plot ablation study results.
    
    Args:
        results: Dictionary of ablation results
        save_path: Optional save path
        metric_name: Name of metric for labels
    """
    # Extract data
    names = list(results.keys())
    metric_diffs = [r.metric_diff for r in results.values()]
    relative_importance = [r.relative_importance for r in results.values()]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Plot 1: Absolute metric difference
    ax1 = axes[0]
    bars = ax1.barh(range(len(names)), metric_diffs, alpha=0.7)
    
    # Color by sign
    colors = ['red' if d < 0 else 'green' for d in metric_diffs]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel(f'{metric_name} Difference (Ablated - Original)', fontsize=12)
    ax1.set_title('Ablation Impact on Performance', fontsize=14, fontweight='bold')
    ax1.axvline(0, color='black', linestyle='--', linewidth=1)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Relative importance
    ax2 = axes[1]
    bars2 = ax2.barh(range(len(names)), relative_importance, alpha=0.7, color='steelblue')
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xlabel('Relative Importance', fontsize=12)
    ax2.set_title('Component Importance (Normalized)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Ablation results saved to {save_path}")
    
    plt.show()
    return fig


def plot_ablation_heatmap(
    results: Dict[str, AblationResult],
    num_layers: int,
    num_heads: int,
    save_path: Optional[str] = None,
    metric: str = 'metric_diff'
):
    """Plot heatmap of ablation results for attention heads.
    
    Args:
        results: Dictionary of ablation results
        num_layers: Number of layers
        num_heads: Number of heads per layer
        save_path: Optional save path
        metric: Which metric to plot ('metric_diff' or 'relative_importance')
    """
    # Create matrix
    matrix = np.zeros((num_layers, num_heads))
    
    for name, result in results.items():
        if result.component_type == 'head':
            # Parse layer and head from name (e.g., 'L0H1')
            try:
                layer = int(name.split('L')[1].split('H')[0])
                head = int(name.split('H')[1])
                
                if metric == 'metric_diff':
                    matrix[layer, head] = result.metric_diff
                elif metric == 'relative_importance':
                    matrix[layer, head] = result.relative_importance
            except:
                continue
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(max(12, num_heads), max(8, num_layers // 2)))
    
    sns.heatmap(
        matrix,
        xticklabels=range(num_heads),
        yticklabels=range(num_layers),
        cmap='RdBu_r' if metric == 'metric_diff' else 'YlOrRd',
        center=0 if metric == 'metric_diff' else None,
        annot=True,
        fmt='.3f',
        cbar_kws={'label': 'Metric Difference' if metric == 'metric_diff' else 'Importance'},
        ax=ax
    )
    
    ax.set_xlabel('Attention Head', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title(f'Ablation Heatmap: {metric.replace("_", " ").title()}', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Ablation heatmap saved to {save_path}")
    
    plt.show()
    return fig


def find_critical_components(
    results: Dict[str, AblationResult],
    threshold: float = 0.1,
    top_k: int = 10
) -> List[AblationResult]:
    """Find most critical components based on ablation results.
    
    Args:
        results: Ablation results dictionary
        threshold: Minimum relative importance threshold
        top_k: Number of top components to return
        
    Returns:
        List of most important ablation results
    """
    # Sort by relative importance
    sorted_results = sorted(
        results.values(),
        key=lambda r: r.relative_importance,
        reverse=True
    )
    
    # Filter by threshold
    critical = [r for r in sorted_results if r.relative_importance >= threshold]
    
    # Return top k
    return critical[:top_k]

