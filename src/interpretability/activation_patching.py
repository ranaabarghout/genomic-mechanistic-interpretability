"""Activation patching for causal analysis of neural networks.

Activation patching (also known as causal tracing or path patching) helps identify
which components of a neural network are causally responsible for specific behaviors.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class PatchingResult:
    """Results from an activation patching experiment."""
    original_output: torch.Tensor
    patched_output: torch.Tensor
    clean_output: torch.Tensor
    corrupted_output: torch.Tensor
    logit_diff: float
    restoration_score: float
    component_name: str
    

class ActivationPatcher:
    """Performs activation patching experiments on neural networks."""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize activation patcher.
        
        Args:
            model: PyTorch model to analyze
            device: Device to run computations on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        self.hooks = []
        self.activations = {}
    
    def _get_activation(self, name: str):
        """Create hook function to capture activations."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                self.activations[name] = output[0].detach()
            else:
                self.activations[name] = output.detach()
        return hook
    
    def _get_patching_hook(self, name: str, patched_activation: torch.Tensor):
        """Create hook function to patch activations."""
        def hook(module, input, output):
            # Replace activation with patched version
            if isinstance(output, tuple):
                output_list = list(output)
                output_list[0] = patched_activation
                return tuple(output_list)
            else:
                return patched_activation
        return hook
    
    def register_hooks(self, layer_names: List[str]):
        """Register hooks to capture activations from specified layers.
        
        Args:
            layer_names: List of layer names to hook
        """
        self.clear_hooks()
        
        for name, module in self.model.named_modules():
            if name in layer_names:
                handle = module.register_forward_hook(self._get_activation(name))
                self.hooks.append(handle)
                print(f"Registered hook on: {name}")
    
    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
    
    def get_output(
        self,
        inputs: torch.Tensor,
        metric_fn: Optional[Callable] = None
    ) -> Tuple[torch.Tensor, float]:
        """Get model output and metric for given inputs.
        
        Args:
            inputs: Input tensor
            metric_fn: Optional function to compute metric from outputs
            
        Returns:
            Tuple of (outputs, metric)
        """
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(inputs)
            
            if isinstance(outputs, tuple):
                logits = outputs[0] if hasattr(outputs[0], 'shape') else outputs.logits
            else:
                logits = outputs if hasattr(outputs, 'shape') else outputs.logits
        
        # Compute metric
        if metric_fn is not None:
            metric = metric_fn(logits)
        else:
            # Default: max logit value
            metric = logits.max().item()
        
        return logits, metric
    
    def patch_activation(
        self,
        clean_input: torch.Tensor,
        corrupted_input: torch.Tensor,
        layer_name: str,
        position_indices: Optional[torch.Tensor] = None,
        metric_fn: Optional[Callable] = None
    ) -> PatchingResult:
        """Patch activations from corrupted run into clean run.
        
        This tests whether replacing a specific activation can restore behavior.
        
        Args:
            clean_input: Clean input that produces desired behavior
            corrupted_input: Corrupted input that doesn't produce desired behavior
            layer_name: Name of layer to patch
            position_indices: Optional indices to patch (for sequence models)
            metric_fn: Function to compute behavioral metric
            
        Returns:
            PatchingResult with outputs and scores
        """
        # 1. Get clean output
        clean_output, clean_metric = self.get_output(clean_input, metric_fn)
        
        # 2. Get corrupted output and save activation
        self.register_hooks([layer_name])
        corrupted_output, corrupted_metric = self.get_output(corrupted_input, metric_fn)
        
        # Extract the activation to patch
        corrupted_activation = self.activations[layer_name].clone()
        self.clear_hooks()
        
        # 3. Run clean input with patched activation
        patched_activation = corrupted_activation.clone()
        
        # Find the module to patch
        target_module = None
        for name, module in self.model.named_modules():
            if name == layer_name:
                target_module = module
                break
        
        if target_module is None:
            raise ValueError(f"Layer {layer_name} not found in model")
        
        # Register patching hook
        handle = target_module.register_forward_hook(
            self._get_patching_hook(layer_name, patched_activation)
        )
        
        # Get patched output
        patched_output, patched_metric = self.get_output(clean_input, metric_fn)
        
        # Remove hook
        handle.remove()
        
        # 4. Compute restoration score
        # How much did patching restore the corrupted behavior?
        logit_diff = corrupted_metric - clean_metric
        patched_diff = patched_metric - clean_metric
        
        if abs(logit_diff) > 1e-6:
            restoration_score = patched_diff / logit_diff
        else:
            restoration_score = 0.0
        
        return PatchingResult(
            original_output=clean_output,
            patched_output=patched_output,
            clean_output=clean_output,
            corrupted_output=corrupted_output,
            logit_diff=logit_diff,
            restoration_score=restoration_score,
            component_name=layer_name
        )
    
    def scan_layers(
        self,
        clean_input: torch.Tensor,
        corrupted_input: torch.Tensor,
        layer_names: List[str],
        metric_fn: Optional[Callable] = None
    ) -> Dict[str, PatchingResult]:
        """Scan multiple layers to find important components.
        
        Args:
            clean_input: Clean input
            corrupted_input: Corrupted input
            layer_names: List of layer names to test
            metric_fn: Metric function
            
        Returns:
            Dictionary mapping layer names to patching results
        """
        results = {}
        
        for layer_name in tqdm(layer_names, desc="Patching layers"):
            try:
                result = self.patch_activation(
                    clean_input,
                    corrupted_input,
                    layer_name,
                    metric_fn=metric_fn
                )
                results[layer_name] = result
            except Exception as e:
                print(f"Error patching {layer_name}: {e}")
                continue
        
        return results


def plot_patching_results(
    results: Dict[str, PatchingResult],
    save_path: Optional[str] = None
):
    """Plot activation patching results.
    
    Args:
        results: Dictionary of patching results
        save_path: Optional path to save plot
    """
    # Extract data
    layer_names = list(results.keys())
    restoration_scores = [r.restoration_score for r in results.values()]
    logit_diffs = [r.logit_diff for r in results.values()]
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Restoration scores
    ax1 = axes[0]
    bars = ax1.barh(range(len(layer_names)), restoration_scores, alpha=0.7)
    
    # Color by magnitude
    colors = plt.cm.RdYlGn(np.array(restoration_scores) / 2 + 0.5)
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax1.set_yticks(range(len(layer_names)))
    ax1.set_yticklabels([name.split('.')[-1] for name in layer_names], fontsize=8)
    ax1.set_xlabel('Restoration Score', fontsize=12)
    ax1.set_title('Activation Patching: Component Importance', fontsize=14, fontweight='bold')
    ax1.axvline(0, color='black', linestyle='--', linewidth=1)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Logit differences
    ax2 = axes[1]
    ax2.barh(range(len(layer_names)), logit_diffs, alpha=0.7, color='steelblue')
    ax2.set_yticks(range(len(layer_names)))
    ax2.set_yticklabels([name.split('.')[-1] for name in layer_names], fontsize=8)
    ax2.set_xlabel('Logit Difference (Corrupted - Clean)', fontsize=12)
    ax2.set_title('Effect of Corruption', fontsize=14, fontweight='bold')
    ax2.axvline(0, color='black', linestyle='--', linewidth=1)
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Patching results saved to {save_path}")
    
    plt.show()
    return fig


def plot_patching_heatmap(
    results: Dict[str, Dict[str, PatchingResult]],
    save_path: Optional[str] = None
):
    """Plot heatmap of patching results across multiple conditions.
    
    Args:
        results: Nested dict {condition: {layer: PatchingResult}}
        save_path: Optional save path
    """
    # Extract conditions and layers
    conditions = list(results.keys())
    layers = list(next(iter(results.values())).keys())
    
    # Create matrix of restoration scores
    matrix = np.zeros((len(conditions), len(layers)))
    
    for i, condition in enumerate(conditions):
        for j, layer in enumerate(layers):
            if layer in results[condition]:
                matrix[i, j] = results[condition][layer].restoration_score
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(16, max(8, len(conditions))))
    
    sns.heatmap(
        matrix,
        xticklabels=[l.split('.')[-1] for l in layers],
        yticklabels=conditions,
        cmap='RdYlGn',
        center=0,
        annot=True,
        fmt='.2f',
        cbar_kws={'label': 'Restoration Score'},
        ax=ax
    )
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Condition', fontsize=12)
    ax.set_title('Activation Patching Heatmap', fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Patching heatmap saved to {save_path}")
    
    plt.show()
    return fig


def create_intervention_plot(
    clean_acts: torch.Tensor,
    corrupted_acts: torch.Tensor,
    layer_name: str,
    save_path: Optional[str] = None
):
    """Visualize activation differences between clean and corrupted runs.
    
    Args:
        clean_acts: Clean activations [seq_len, hidden_dim]
        corrupted_acts: Corrupted activations [seq_len, hidden_dim]
        layer_name: Name of layer
        save_path: Optional save path
    """
    # Compute difference
    diff = (corrupted_acts - clean_acts).cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Clean activations
    im1 = axes[0].imshow(clean_acts.cpu().numpy().T, aspect='auto', cmap='viridis')
    axes[0].set_title('Clean Activations')
    axes[0].set_xlabel('Position')
    axes[0].set_ylabel('Hidden Dimension')
    plt.colorbar(im1, ax=axes[0])
    
    # Plot 2: Corrupted activations
    im2 = axes[1].imshow(corrupted_acts.cpu().numpy().T, aspect='auto', cmap='viridis')
    axes[1].set_title('Corrupted Activations')
    axes[1].set_xlabel('Position')
    plt.colorbar(im2, ax=axes[1])
    
    # Plot 3: Difference
    im3 = axes[2].imshow(diff.T, aspect='auto', cmap='RdBu_r', vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
    axes[2].set_title('Difference (Corrupted - Clean)')
    axes[2].set_xlabel('Position')
    plt.colorbar(im3, ax=axes[2])
    
    plt.suptitle(f'Activation Analysis: {layer_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Intervention plot saved to {save_path}")
    
    plt.show()
    return fig

