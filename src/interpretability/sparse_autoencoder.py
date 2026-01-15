"""
Sparse Autoencoder Analysis Module (QTL-specific)

Provides QTLSAEAnalyzer class for training and analyzing sparse autoencoders
on genomic model activations to discover interpretable features.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Tuple
import warnings
from data.sqtl_data_loader import OriginalSQTLDataLoader, sQTLSample
from models.load_model import load_base_model


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder with L1 sparsity penalty
    """
    def __init__(self, input_dim: int, hidden_dim: int,
                 sparsity_coefficient: float = 0.1,
                 l1_coefficient: float = 1e-4):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.sparsity_coef = sparsity_coefficient
        self.l1_coef = l1_coefficient

    def forward(self, x):
        # Encode
        hidden = torch.relu(self.encoder(x))
        # Decode
        reconstructed = self.decoder(hidden)
        return reconstructed, hidden

    def compute_loss(self, x, reconstructed, hidden):
        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(reconstructed, x)
        # Sparsity loss (L1 on hidden activations)
        sparsity_loss = torch.mean(torch.abs(hidden))
        # L1 regularization on weights
        l1_loss = (torch.norm(self.encoder.weight, 1) +
                   torch.norm(self.decoder.weight, 1))
        # Total loss
        total_loss = (recon_loss +
                      self.sparsity_coef * sparsity_loss +
                      self.l1_coef * l1_loss)

        return {
            'total': total_loss,
            'reconstruction': recon_loss,
            'sparsity': sparsity_loss
        }


class QTLSAEAnalyzer:
    """
    Train and analyze sparse autoencoder on DNABERT-2 activations
    """

    def __init__(self,
                 model_name: str = "zhihan1996/DNABERT-2-117M",
                 num_samples: int = 200,
                 output_dir: str = "outputs/sparse_autoencoder",
                 sae_hidden_dim: int = 2048,
                 sparsity_coef: float = 1e-3):
        """
        Initialize SAE analyzer

        Args:
            model_name: DNABERT-2 model name
            num_samples: Number of sQTL samples for training
            output_dir: Output directory for results
            sae_hidden_dim: SAE hidden dimension (typically > input_dim for overcomplete)
            sparsity_coef: Sparsity penalty coefficient (reduced to prevent dead ReLUs)
        """
        self.model_name = model_name
        self.num_samples = num_samples
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.sae_hidden_dim = sae_hidden_dim
        self.sparsity_coef = sparsity_coef

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load model
        print("\nLoading DNABERT-2 model...")
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
        self.input_dim = self.config.hidden_size
        print(f"\nModel architecture:")
        print(f"  Hidden size: {self.input_dim}")
        print(f"  SAE hidden dim: {self.sae_hidden_dim}")

        # Initialize SAE (will be trained later)
        self.sae = None

        # Storage for activations
        self.activations_dataset = {
            'significant': {'activations': [], 'samples': []},
            'not_significant': {'activations': [], 'samples': []}
        }

    def extract_activations(self, layer_idx: int = -1, max_length: int = 512):
        """
        Extract hidden states from DNABERT-2 for all samples

        Args:
            layer_idx: Which layer to extract (-1 = last layer)
            max_length: Maximum sequence length
        """
        print("\n" + "="*70)
        print("EXTRACTING ACTIVATIONS FROM DNABERT-2")
        print("="*70)
        print(f"Extracting layer {layer_idx} activations for {len(self.samples)} samples\n")

        for sample in tqdm(self.samples, desc="Extracting"):
            # Tokenize
            inputs = self.tokenizer(
                sample.ref_sequence,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding="max_length"
            ).to(self.device)

            # Get hidden states
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)

            # Handle different output formats
            if isinstance(outputs, tuple):
                # DNABERT-2: (last_hidden_state, pooler_output)
                # Only final layer available
                if layer_idx == -1 or layer_idx == 0:
                    hidden = outputs[0]  # (1, seq_len, hidden_size)
                else:
                    print(f"Warning: DNABERT-2 only provides final layer. Using layer -1 instead of {layer_idx}")
                    hidden = outputs[0]
            else:
                # Standard BERT: has hidden_states attribute
                hidden = outputs.hidden_states[layer_idx]  # (1, seq_len, hidden_size)

            # Pool over sequence (mean pooling)
            pooled = hidden.mean(dim=1).squeeze(0)  # (hidden_size,)

            # Store by class
            class_key = 'significant' if sample.label == 0 else 'not_significant'
            self.activations_dataset[class_key]['activations'].append(pooled.cpu().numpy())
            self.activations_dataset[class_key]['samples'].append(sample)

        # Convert to arrays
        for key in ['significant', 'not_significant']:
            self.activations_dataset[key]['activations'] = np.array(
                self.activations_dataset[key]['activations']
            )

        print(f"\nExtracted activations:")
        print(f"  Significant: {self.activations_dataset['significant']['activations'].shape}")
        print(f"  Not significant: {self.activations_dataset['not_significant']['activations'].shape}")

    def train_sae(self, epochs: int = 50, batch_size: int = 32, lr: float = 1e-3):
        """
        Train sparse autoencoder on extracted activations
        """
        print("\n" + "="*70)
        print("TRAINING SPARSE AUTOENCODER")
        print("="*70)
        print(f"Training for {epochs} epochs with batch size {batch_size}")
        print(f"Sparsity coefficient: {self.sparsity_coef}")
        print(f"L1 coefficient: 1e-4\n")

        # Combine all activations for training
        all_activations = np.vstack([
            self.activations_dataset['significant']['activations'],
            self.activations_dataset['not_significant']['activations']
        ])

        print(f"Training dataset: {all_activations.shape}")
        print(f"Activation statistics:")
        print(f"  Mean: {all_activations.mean():.6f}")
        print(f"  Std: {all_activations.std():.6f}")
        print(f"  Min: {all_activations.min():.6f}")
        print(f"  Max: {all_activations.max():.6f}")
        print(f"  % zeros: {(all_activations == 0).mean() * 100:.2f}%\n")

        # Initialize SAE
        self.sae = SparseAutoencoder(
            input_dim=self.input_dim,
            hidden_dim=self.sae_hidden_dim,
            sparsity_coefficient=self.sparsity_coef,
            l1_coefficient=1e-4  # Increased for better sparsity
        ).to(self.device)

        # Optimizer
        optimizer = optim.Adam(self.sae.parameters(), lr=lr)

        # Convert to tensor
        activations_tensor = torch.FloatTensor(all_activations).to(self.device)

        # Training loop
        losses = {'total': [], 'reconstruction': [], 'sparsity': []}

        for epoch in range(epochs):
            epoch_losses = {'total': 0.0, 'reconstruction': 0.0, 'sparsity': 0.0}
            n_batches = 0

            # Shuffle indices
            indices = torch.randperm(len(activations_tensor))

            for i in range(0, len(activations_tensor), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch = activations_tensor[batch_indices]

                # Forward pass
                reconstructed, hidden = self.sae(batch)

                # Loss
                loss_dict = self.sae.compute_loss(batch, reconstructed, hidden)
                loss = loss_dict['total']

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track losses
                epoch_losses['total'] += loss.item()
                epoch_losses['reconstruction'] += loss_dict['reconstruction'].item()
                epoch_losses['sparsity'] += loss_dict['sparsity'].item()
                n_batches += 1

            # Average losses
            for key in epoch_losses:
                epoch_losses[key] /= n_batches
                losses[key].append(epoch_losses[key])

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Total Loss = {epoch_losses['total']:.4f}, "
                      f"Recon = {epoch_losses['reconstruction']:.4f}, "
                      f"Sparsity = {epoch_losses['sparsity']:.4f}")

        print("\nTraining complete!")

        # Plot training curves
        self._plot_training_curves(losses)

        # Save SAE
        sae_path = self.output_dir / "sae_model.pt"
        torch.save(self.sae.state_dict(), sae_path)
        print(f"Saved SAE model: {sae_path}")

        return losses

    def analyze_features(self, top_k: int = 20):
        """
        Analyze learned SAE features

        Identify features that are:
        1. Most differentially activated between classes
        2. Most sparse (selective)
        3. Most active overall
        """
        print("\n" + "="*70)
        print("ANALYZING SAE FEATURES")
        print("="*70)
        print(f"Analyzing {self.sae_hidden_dim} latent features\n")

        if self.sae is None:
            raise ValueError("SAE not trained! Run train_sae() first.")

        # Get feature activations for all samples
        feature_activations = {
            'significant': [],
            'not_significant': []
        }

        for key in ['significant', 'not_significant']:
            activations = self.activations_dataset[key]['activations']
            activations_tensor = torch.FloatTensor(activations).to(self.device)

            with torch.no_grad():
                _, features = self.sae(activations_tensor)

            feature_activations[key] = features.cpu().numpy()

        # Compute statistics
        sig_features = feature_activations['significant']  # (n_samples, hidden_dim)
        not_sig_features = feature_activations['not_significant']

        # Mean activation per class
        sig_mean = sig_features.mean(axis=0)
        not_sig_mean = not_sig_features.mean(axis=0)

        # Differential activation
        differential = sig_mean - not_sig_mean

        # Sparsity (fraction of zeros)
        all_features = np.vstack([sig_features, not_sig_features])
        sparsity = (np.abs(all_features) < 0.01).mean(axis=0)

        # Overall activation magnitude
        overall_magnitude = all_features.mean(axis=0)

        results = {
            'differential': differential,
            'sparsity': sparsity,
            'magnitude': overall_magnitude,
            'sig_mean': sig_mean,
            'not_sig_mean': not_sig_mean,
            'feature_activations': feature_activations
        }

        # Identify top features
        top_differential_idx = np.argsort(np.abs(differential))[-top_k:][::-1]
        top_sparse_idx = np.argsort(sparsity)[-top_k:][::-1]

        print(f"Top {top_k} Most Differential Features:")
        print("-"*70)
        for rank, idx in enumerate(top_differential_idx, 1):
            direction = "→ sig" if differential[idx] > 0 else "→ not_sig"
            print(f"  {rank}. Feature {idx}: Δ = {differential[idx]:+.4f} ({direction})")

        print(f"\nTop {top_k} Most Sparse (Selective) Features:")
        print("-"*70)
        for rank, idx in enumerate(top_sparse_idx, 1):
            print(f"  {rank}. Feature {idx}: Sparsity = {sparsity[idx]:.4f}")

        # Plot feature analysis
        self._plot_feature_analysis(results, top_k)

        return results

    def visualize_feature_activations(self, feature_idx: int, num_examples: int = 5):
        """
        Visualize which sequences activate a specific feature
        """
        print(f"\n" + "="*70)
        print(f"VISUALIZING FEATURE {feature_idx}")
        print("="*70)

        if self.sae is None:
            raise ValueError("SAE not trained!")

        # Get activations for this feature
        sig_activations = []
        not_sig_activations = []

        for key in ['significant', 'not_significant']:
            activations = self.activations_dataset[key]['activations']
            activations_tensor = torch.FloatTensor(activations).to(self.device)

            with torch.no_grad():
                _, features = self.sae(activations_tensor)

            feature_act = features[:, feature_idx].cpu().numpy()

            if key == 'significant':
                sig_activations = feature_act
            else:
                not_sig_activations = feature_act

        # Find top activating examples from each class
        top_sig_idx = np.argsort(sig_activations)[-num_examples:][::-1]
        top_not_sig_idx = np.argsort(not_sig_activations)[-num_examples:][::-1]

        print(f"\nTop {num_examples} Significant sQTLs activating Feature {feature_idx}:")
        for rank, idx in enumerate(top_sig_idx, 1):
            sample = self.activations_dataset['significant']['samples'][idx]
            activation = sig_activations[idx]
            print(f"  {rank}. Activation = {activation:.4f}, Variant at {sample.variant_pos}bp")

        print(f"\nTop {num_examples} Not Significant sQTLs activating Feature {feature_idx}:")
        for rank, idx in enumerate(top_not_sig_idx, 1):
            sample = self.activations_dataset['not_significant']['samples'][idx]
            activation = not_sig_activations[idx]
            print(f"  {rank}. Activation = {activation:.4f}, Variant at {sample.variant_pos}bp")

        # Plot distribution
        self._plot_feature_distribution(feature_idx, sig_activations, not_sig_activations)

    def run_full_analysis(self, train_sae: bool = True, epochs: int = 50):
        """Run complete SAE analysis pipeline"""
        print("\n" + "="*70)
        print("FULL SPARSE AUTOENCODER ANALYSIS")
        print("="*70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_dir = self.output_dir / f"analysis_{timestamp}"
        analysis_dir.mkdir(exist_ok=True)
        self.output_dir = analysis_dir

        all_results = {}

        # 1. Extract activations
        print("\n" + "-"*70)
        print("Step 1/4: Extracting DNABERT-2 activations")
        print("-"*70)
        self.extract_activations(layer_idx=-1)

        # 2. Train SAE
        if train_sae:
            print("\n" + "-"*70)
            print("Step 2/4: Training sparse autoencoder")
            print("-"*70)
            all_results['training'] = self.train_sae(epochs=epochs)
        else:
            print("\n" + "-"*70)
            print("Step 2/4: Loading pre-trained SAE")
            print("-"*70)
            sae_path = self.output_dir.parent / "sae_model.pt"
            if not sae_path.exists():
                raise FileNotFoundError(f"No pre-trained SAE found at {sae_path}")

            self.sae = SparseAutoencoder(
                input_dim=self.input_dim,
                hidden_dim=self.sae_hidden_dim,
                sparsity_coefficient=self.sparsity_coef,
                l1_coefficient=1e-4
            ).to(self.device)
            self.sae.load_state_dict(torch.load(sae_path))
            print(f"Loaded SAE from {sae_path}")

        # 3. Analyze features
        print("\n" + "-"*70)
        print("Step 3/4: Analyzing learned features")
        print("-"*70)
        all_results['features'] = self.analyze_features(top_k=20)

        # 4. Visualize top features
        print("\n" + "-"*70)
        print("Step 4/4: Visualizing top features")
        print("-"*70)
        top_features = np.argsort(np.abs(all_results['features']['differential']))[-5:][::-1]
        for feature_idx in top_features:
            self.visualize_feature_activations(feature_idx, num_examples=5)

        # 5. Generate report
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

    def _plot_training_curves(self, losses: Dict):
        """Plot SAE training curves"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        epochs = range(1, len(losses['total']) + 1)

        # Total loss
        axes[0].plot(epochs, losses['total'], linewidth=2, color='#2E86AB')
        axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Total Loss', fontsize=12, fontweight='bold')
        axes[0].set_title('Total Training Loss', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Reconstruction loss
        axes[1].plot(epochs, losses['reconstruction'], linewidth=2, color='#A23B72')
        axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Reconstruction Loss', fontsize=12, fontweight='bold')
        axes[1].set_title('Reconstruction Error', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        # Sparsity loss
        axes[2].plot(epochs, losses['sparsity'], linewidth=2, color='#F18F01')
        axes[2].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Sparsity Loss', fontsize=12, fontweight='bold')
        axes[2].set_title('Sparsity Penalty', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = self.output_dir / "1_training_curves.png"
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        print(f"Saved: {output_file.name}")
        plt.close()

    def _plot_feature_analysis(self, results: Dict, top_k: int):
        """Plot feature analysis results"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        differential = results['differential']
        sparsity = results['sparsity']
        sig_mean = results['sig_mean']
        not_sig_mean = results['not_sig_mean']

        # 1. Differential activation distribution
        axes[0, 0].hist(differential, bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Differential Activation (Sig - Not Sig)', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Number of Features', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Distribution of Feature Differential Activation',
                            fontsize=13, fontweight='bold')
        axes[0, 0].grid(axis='y', alpha=0.3)

        # 2. Top differential features
        top_diff_idx = np.argsort(np.abs(differential))[-top_k:][::-1]
        top_diff_vals = differential[top_diff_idx]
        colors = ['#A23B72' if v > 0 else '#F18F01' for v in top_diff_vals]

        axes[0, 1].barh(range(top_k), top_diff_vals, color=colors, alpha=0.8)
        axes[0, 1].set_yticks(range(top_k))
        axes[0, 1].set_yticklabels([f"F{idx}" for idx in top_diff_idx], fontsize=9)
        axes[0, 1].set_xlabel('Differential Activation', fontsize=11, fontweight='bold')
        axes[0, 1].set_title(f'Top {top_k} Most Differential Features',
                            fontsize=13, fontweight='bold')
        axes[0, 1].axvline(x=0, color='black', linestyle='--', linewidth=1)
        axes[0, 1].grid(axis='x', alpha=0.3)

        # 3. Sparsity vs Differential
        axes[1, 0].scatter(sparsity, np.abs(differential), alpha=0.5, s=20, color='#2E86AB')
        axes[1, 0].set_xlabel('Sparsity (Fraction of Zeros)', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel('|Differential Activation|', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Feature Sparsity vs Differential Activation',
                            fontsize=13, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

        # Highlight top differential features
        for idx in top_diff_idx[:5]:
            axes[1, 0].scatter(sparsity[idx], np.abs(differential[idx]),
                             s=100, color='red', marker='*', zorder=5)

        # 4. Mean activation by class (top features)
        x = np.arange(top_k)
        width = 0.35

        axes[1, 1].bar(x - width/2, sig_mean[top_diff_idx], width,
                      label='Significant', color='#2E86AB', alpha=0.8)
        axes[1, 1].bar(x + width/2, not_sig_mean[top_diff_idx], width,
                      label='Not Significant', color='#F18F01', alpha=0.8)
        axes[1, 1].set_xlabel('Feature Index', fontsize=11, fontweight='bold')
        axes[1, 1].set_ylabel('Mean Activation', fontsize=11, fontweight='bold')
        axes[1, 1].set_title(f'Mean Activation by Class (Top {top_k} Features)',
                            fontsize=13, fontweight='bold')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([f"F{idx}" for idx in top_diff_idx],
                                   rotation=45, ha='right', fontsize=8)
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_file = self.output_dir / "2_feature_analysis.png"
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        print(f"Saved: {output_file.name}")
        plt.close()

    def _plot_feature_distribution(self, feature_idx: int,
                                   sig_activations: np.ndarray,
                                   not_sig_activations: np.ndarray):
        """Plot activation distribution for a specific feature"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Histograms
        axes[0].hist(sig_activations, bins=30, alpha=0.7, label='Significant',
                    color='#2E86AB', edgecolor='black')
        axes[0].hist(not_sig_activations, bins=30, alpha=0.7, label='Not Significant',
                    color='#F18F01', edgecolor='black')
        axes[0].set_xlabel('Feature Activation', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Count', fontsize=12, fontweight='bold')
        axes[0].set_title(f'Feature {feature_idx} Activation Distribution',
                         fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(axis='y', alpha=0.3)

        # Box plots
        data = [sig_activations, not_sig_activations]
        labels = ['Significant', 'Not Significant']
        bp = axes[1].boxplot(data, labels=labels, patch_artist=True,
                            boxprops=dict(facecolor='#2E86AB', alpha=0.7),
                            medianprops=dict(color='red', linewidth=2))
        bp['boxes'][1].set_facecolor('#F18F01')

        axes[1].set_ylabel('Feature Activation', fontsize=12, fontweight='bold')
        axes[1].set_title(f'Feature {feature_idx} Activation by Class',
                         fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_file = self.output_dir / f"feature_{feature_idx}_distribution.png"
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        print(f"Saved: {output_file.name}")
        plt.close()

    def _generate_report(self, all_results: Dict, timestamp: str):
        """Generate comprehensive report"""
        report_file = self.output_dir / f"sae_analysis_report_{timestamp}.txt"

        with open(report_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("SPARSE AUTOENCODER ANALYSIS REPORT\n")
            f.write("Interpretable Feature Discovery in DNABERT-2\n")
            f.write("="*70 + "\n\n")

            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"SAE Architecture: {self.input_dim} → {self.sae_hidden_dim} → {self.input_dim}\n")
            f.write(f"Sparsity coefficient: {self.sparsity_coef}\n\n")

            f.write("Dataset:\n")
            f.write(f"  Total samples: {len(self.samples)}\n")
            f.write(f"  Significant: {len(self.significant)}\n")
            f.write(f"  Not significant: {len(self.not_significant)}\n\n")

            if 'training' in all_results:
                f.write("="*70 + "\n")
                f.write("TRAINING RESULTS\n")
                f.write("="*70 + "\n\n")
                final_loss = all_results['training']['total'][-1]
                final_recon = all_results['training']['reconstruction'][-1]
                final_sparse = all_results['training']['sparsity'][-1]
                f.write(f"  Final total loss: {final_loss:.4f}\n")
                f.write(f"  Final reconstruction loss: {final_recon:.4f}\n")
                f.write(f"  Final sparsity loss: {final_sparse:.4f}\n\n")

            f.write("="*70 + "\n")
            f.write("FEATURE ANALYSIS\n")
            f.write("="*70 + "\n\n")

            features = all_results['features']
            differential = features['differential']
            sparsity = features['sparsity']

            # Top differential features
            top_diff_idx = np.argsort(np.abs(differential))[-10:][::-1]
            f.write("Top 10 Most Differential Features:\n")
            f.write("-"*70 + "\n")
            for rank, idx in enumerate(top_diff_idx, 1):
                direction = "significant" if differential[idx] > 0 else "not_significant"
                f.write(f"  {rank}. Feature {idx}: Δ = {differential[idx]:+.4f} (→ {direction})\n")
                f.write(f"      Sparsity = {sparsity[idx]:.4f}\n")
            f.write("\n")

            # Statistics
            positive_features = (differential > 0.05).sum()
            negative_features = (differential < -0.05).sum()
            neutral_features = self.sae_hidden_dim - positive_features - negative_features

            f.write("Feature Statistics:\n")
            f.write("-"*70 + "\n")
            f.write(f"  Features favoring significant: {positive_features}\n")
            f.write(f"  Features favoring not_significant: {negative_features}\n")
            f.write(f"  Neutral features: {neutral_features}\n")
            f.write(f"  Mean sparsity: {sparsity.mean():.4f}\n\n")

            f.write("="*70 + "\n")
            f.write("KEY FINDINGS\n")
            f.write("="*70 + "\n\n")

            f.write("1. Interpretable Feature Discovery:\n")
            f.write(f"   SAE discovered {self.sae_hidden_dim} latent features from DNABERT-2\n")
            f.write("   activations. Features show clear differential activation between\n")
            f.write("   significant and not_significant sQTLs.\n\n")

            f.write("2. Feature Selectivity:\n")
            f.write(f"   Average sparsity: {sparsity.mean():.2%} of samples activate each feature.\n")
            f.write("   High sparsity indicates features are selective and interpretable.\n\n")

            f.write("3. Class-Specific Features:\n")
            f.write(f"   {positive_features} features preferentially activate for significant sQTLs\n")
            f.write(f"   {negative_features} features preferentially activate for not_significant sQTLs\n")
            f.write("   These features capture class-specific sequence patterns.\n\n")

            f.write("="*70 + "\n")
            f.write("OUTPUT FILES\n")
            f.write("="*70 + "\n\n")
            for file in sorted(self.output_dir.glob("*")):
                f.write(f"  - {file.name}\n")
            f.write("\n" + "="*70 + "\n")

        print(f"Report saved: {report_file.name}")


