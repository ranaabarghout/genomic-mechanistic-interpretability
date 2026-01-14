"""Sparse autoencoder for latent feature discovery in genomic models."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, Dict, List
from tqdm import tqdm


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder for discovering interpretable latent features.
    
    SAEs can help identify monosemantic features in neural network activations
    by enforcing sparsity in the latent representation.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        sparsity_coefficient: float = 1e-3,
        l1_coefficient: float = 1e-4
    ):
        """Initialize Sparse Autoencoder.
        
        Args:
            input_dim: Dimension of input activations
            hidden_dim: Dimension of sparse latent space (typically larger than input)
            sparsity_coefficient: Weight for sparsity loss
            l1_coefficient: L1 regularization coefficient
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_coefficient = sparsity_coefficient
        self.l1_coefficient = l1_coefficient
        
        # Encoder
        self.encoder = nn.Linear(input_dim, hidden_dim)
        
        # Decoder
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.encoder.weight)
        # Initialize bias to small positive value to prevent dead ReLUs
        nn.init.constant_(self.encoder.bias, 0.1)
        nn.init.xavier_uniform_(self.decoder.weight)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse latent representation.
        
        Args:
            x: Input tensor [batch, input_dim]
            
        Returns:
            Sparse latent codes [batch, hidden_dim]
        """
        return F.relu(self.encoder(x))
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction.
        
        Args:
            z: Latent codes [batch, hidden_dim]
            
        Returns:
            Reconstructed input [batch, input_dim]
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through autoencoder.
        
        Args:
            x: Input tensor [batch, input_dim]
            
        Returns:
            Tuple of (reconstructed input, latent codes)
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    
    def compute_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        z: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute SAE loss with sparsity penalty.
        
        Args:
            x: Original input
            x_recon: Reconstructed input
            z: Latent codes
            
        Returns:
            Dictionary with loss components
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x)
        
        # L1 sparsity penalty on activations
        l1_loss = self.l1_coefficient * torch.abs(z).mean()
        
        # Simple sparsity penalty: penalize high activation rates
        # More numerically stable than KL divergence for ReLU activations
        sparsity_target = 0.05
        mean_activation = z.mean(dim=0)
        # L2 penalty on deviation from target sparsity
        sparsity_loss = self.sparsity_coefficient * torch.mean((mean_activation - sparsity_target) ** 2)
        
        # Total loss
        total_loss = recon_loss + l1_loss + sparsity_loss
        
        return {
            'total': total_loss,
            'reconstruction': recon_loss,
            'l1': l1_loss,
            'sparsity': sparsity_loss
        }


def train_sae(
    sae: SparseAutoencoder,
    activations: torch.Tensor,
    num_epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    verbose: bool = True
) -> Dict[str, List[float]]:
    """Train Sparse Autoencoder on model activations.
    
    Args:
        sae: SparseAutoencoder instance
        activations: Activation tensors to train on [num_samples, input_dim]
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for Adam optimizer
        device: Device to train on
        verbose: Whether to print progress
        
    Returns:
        Dictionary with training history
    """
    sae = sae.to(device)
    sae.train()
    
    # Create dataloader
    dataset = TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.Adam(sae.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'total_loss': [],
        'recon_loss': [],
        'l1_loss': [],
        'sparsity_loss': [],
        'sparsity': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_losses = {k: [] for k in history.keys()}
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}') if verbose else dataloader
        
        for (batch,) in pbar:
            batch = batch.to(device)
            
            # Forward pass
            x_recon, z = sae(batch)
            
            # Compute losses
            losses = sae.compute_loss(batch, x_recon, z)
            
            # Backward pass
            optimizer.zero_grad()
            losses['total'].backward()
            optimizer.step()
            
            # Record losses
            epoch_losses['total_loss'].append(losses['total'].item())
            epoch_losses['recon_loss'].append(losses['reconstruction'].item())
            epoch_losses['l1_loss'].append(losses['l1'].item())
            epoch_losses['sparsity_loss'].append(losses['sparsity'].item())
            epoch_losses['sparsity'].append((z > 0).float().mean().item())
            
            if verbose and isinstance(pbar, tqdm):
                pbar.set_postfix({
                    'loss': losses['total'].item(),
                    'recon': losses['reconstruction'].item(),
                    'sparsity': epoch_losses['sparsity'][-1]
                })
        
        # Store epoch averages
        for key in history.keys():
            history[key].append(np.mean(epoch_losses[key]))
        
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Loss: {history['total_loss'][-1]:.4f}, "
                  f"Recon: {history['recon_loss'][-1]:.4f}, "
                  f"Sparsity: {history['sparsity'][-1]:.4f}")
    
    return history


def extract_activations(
    model,
    dataloader: DataLoader,
    layer_name: str,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    max_samples: Optional[int] = None
) -> torch.Tensor:
    """Extract activations from a specific layer of the model.
    
    Args:
        model: Neural network model
        dataloader: DataLoader with input data
        layer_name: Name of layer to extract activations from
        device: Device to run model on
        max_samples: Maximum number of samples to extract
        
    Returns:
        Tensor of activations [num_samples, activation_dim]
    """
    model.eval()
    model.to(device)
    
    activations = []
    hooks = []
    
    # Register hook to capture activations
    def hook_fn(module, input, output):
        # Handle different output types
        if isinstance(output, tuple):
            act = output[0]
        else:
            act = output
        
        # Flatten if needed
        if len(act.shape) > 2:
            act = act.reshape(act.size(0), -1)
        
        activations.append(act.detach().cpu())
    
    # Find and register hook on target layer
    for name, module in model.named_modules():
        if layer_name in name:
            handle = module.register_forward_hook(hook_fn)
            hooks.append(handle)
            print(f"Registered hook on: {name}")
            break
    
    # Extract activations
    num_samples = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting activations"):
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch
            
            inputs = inputs.to(device)
            _ = model(inputs)
            
            num_samples += inputs.size(0)
            if max_samples and num_samples >= max_samples:
                break
    
    # Remove hooks
    for handle in hooks:
        handle.remove()
    
    # Concatenate all activations
    all_activations = torch.cat(activations, dim=0)
    
    if max_samples:
        all_activations = all_activations[:max_samples]
    
    print(f"Extracted {all_activations.shape[0]} activation vectors of dimension {all_activations.shape[1]}")
    
    return all_activations


def analyze_learned_features(
    sae: SparseAutoencoder,
    activations: torch.Tensor,
    top_k: int = 20,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict:
    """Analyze features learned by SAE.
    
    Args:
        sae: Trained SparseAutoencoder
        activations: Activations to analyze
        top_k: Number of top features to analyze
        device: Device to run on
        
    Returns:
        Dictionary with feature analysis
    """
    sae.eval()
    sae.to(device)
    activations = activations.to(device)
    
    with torch.no_grad():
        # Encode activations
        z = sae.encode(activations)
        
        # Compute feature statistics
        feature_means = z.mean(dim=0)
        feature_stds = z.std(dim=0)
        feature_max = z.max(dim=0)[0]
        
        # Compute sparsity (fraction of zeros)
        sparsity = (z == 0).float().mean(dim=0)
        
        # Find most active features
        activation_frequency = (z > 0).float().mean(dim=0)
        top_features = torch.argsort(activation_frequency, descending=True)[:top_k]
    
    results = {
        'latent_codes': z.cpu(),
        'feature_means': feature_means.cpu(),
        'feature_stds': feature_stds.cpu(),
        'feature_max': feature_max.cpu(),
        'sparsity': sparsity.cpu(),
        'activation_frequency': activation_frequency.cpu(),
        'top_features': top_features.cpu(),
        'decoder_weights': sae.decoder.weight.cpu()
    }
    
    return results


def plot_sae_analysis(
    results: Dict,
    history: Dict,
    save_path: Optional[str] = None
):
    """Create comprehensive plots for SAE analysis.
    
    Args:
        results: Results from analyze_learned_features
        history: Training history from train_sae
        save_path: Optional path to save plots
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Training curves
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(history['total_loss'], label='Total Loss', linewidth=2)
    ax1.plot(history['recon_loss'], label='Reconstruction Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Sparsity over training
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(history['sparsity'], linewidth=2, color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Sparsity')
    ax2.set_title('Activation Sparsity')
    ax2.grid(True, alpha=0.3)
    
    # 3. Feature activation frequency
    ax3 = fig.add_subplot(gs[1, 0])
    freq = results['activation_frequency'].numpy()
    ax3.hist(freq, bins=50, edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Activation Frequency')
    ax3.set_ylabel('Count')
    ax3.set_title('Feature Activation Distribution')
    ax3.axvline(freq.mean(), color='red', linestyle='--', label=f'Mean: {freq.mean():.3f}')
    ax3.legend()
    
    # 4. Top features
    ax4 = fig.add_subplot(gs[1, 1])
    top_k = min(20, len(results['top_features']))
    top_indices = results['top_features'][:top_k].numpy()
    top_freqs = results['activation_frequency'][top_indices].numpy()
    ax4.barh(range(top_k), top_freqs, alpha=0.7)
    ax4.set_yticks(range(top_k))
    ax4.set_yticklabels([f'Feature {i}' for i in top_indices])
    ax4.set_xlabel('Activation Frequency')
    ax4.set_title(f'Top {top_k} Most Active Features')
    ax4.invert_yaxis()
    
    # 5. Decoder weight norms
    ax5 = fig.add_subplot(gs[1, 2])
    weight_norms = torch.norm(results['decoder_weights'], dim=1).numpy()
    ax5.hist(weight_norms, bins=50, edgecolor='black', alpha=0.7)
    ax5.set_xlabel('Weight Norm')
    ax5.set_ylabel('Count')
    ax5.set_title('Decoder Weight Magnitudes')
    ax5.axvline(weight_norms.mean(), color='red', linestyle='--', label=f'Mean: {weight_norms.mean():.2f}')
    ax5.legend()
    
    # 6. Latent code visualization (first 2 PCs or t-SNE)
    ax6 = fig.add_subplot(gs[2, :])
    z = results['latent_codes'].numpy()
    
    # Random sample for visualization
    sample_size = min(1000, z.shape[0])
    indices = np.random.choice(z.shape[0], sample_size, replace=False)
    z_sample = z[indices]
    
    # Simple 2D projection (first two dimensions)
    ax6.scatter(z_sample[:, 0], z_sample[:, 1], alpha=0.5, s=10)
    ax6.set_xlabel('Latent Dimension 1')
    ax6.set_ylabel('Latent Dimension 2')
    ax6.set_title('Latent Space Visualization (First 2 Dimensions)')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Sparse Autoencoder Analysis', fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SAE analysis plot saved to {save_path}")
    
    plt.show()
    return fig

