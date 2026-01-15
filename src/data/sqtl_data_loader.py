"""
sQTL Data Loader - Load Original DNA Sequences
===============================================
Loads original sQTL DNA sequences from genomic-FM data wrapper
instead of PCA-transformed embeddings.

This enables full interpretability analysis including:
- Attention visualization (which nucleotides matter)
- Gradient-based saliency maps
- Position-specific importance
"""

import sys
import os
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass

# Add genomic-FM to path - need both root (for package) and src (for direct imports)
GENOMIC_FM_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "genomic-FM"
GENOMIC_FM_SRC = GENOMIC_FM_ROOT / "src"
for path in [str(GENOMIC_FM_ROOT), str(GENOMIC_FM_SRC)]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Import as package to support relative imports in genomic-FM
from src.dataloader.data_wrapper import sQTLDataWrapper


@dataclass
class sQTLSample:
    """Single sQTL sample with original sequences"""
    ref_sequence: str  # Reference DNA sequence (1024bp)
    alt_sequence: str  # Alternate DNA sequence (1024bp)
    tissue: str        # Tissue type (e.g., "Whole_Blood")
    label: int         # 0=significant, 1=not_significant
    label_name: str    # "significant" or "not_significant"
    variant_position: int  # Position where ref != alt

    @property
    def variant_pos(self):
        """Alias for variant_position (for compatibility)"""
        return self.variant_position

    def __repr__(self):
        return (f"sQTLSample(tissue={self.tissue}, label={self.label_name}, "
                f"variant_pos={self.variant_position}, seq_len={len(self.ref_sequence)})")


class OriginalSQTLDataLoader:
    """Load original sQTL DNA sequences for interpretability analysis"""

    def __init__(self,
                 num_samples: int = None,
                 num_records: int = None,  # Alias for compatibility
                 all_records: bool = False,
                 seq_length: int = 1024,
                 tissue_filter: str = None):
        """
        Initialize data loader

        Args:
            num_samples: Number of samples to load (None = all)
            num_records: Alias for num_samples (for compatibility)
            all_records: If True, load all available records
            seq_length: DNA sequence length (default: 1024bp)
            tissue_filter: Filter by tissue type (e.g., "Whole_Blood")
        """
        # Accept either num_samples or num_records
        self.num_samples = num_samples or num_records
        self.all_records = all_records
        self.seq_length = seq_length
        self.tissue_filter = tissue_filter

        # Label mapping (from genomic-FM training)
        self.label_map = {
            "positive": 0,      # significant sQTL
            "negative": 1,      # not significant
            "significant": 0,
            "not_significant": 1
        }
        self.label_names = {0: "significant", 1: "not_significant"}

        # Store loaded samples
        self.samples = []

        print(f"Initializing sQTL data loader...")
        print(f"  Sequence length: {seq_length}bp")
        print(f"  Tissue filter: {tissue_filter or 'All'}")

    def load_data(self) -> List[sQTLSample]:
        """
        Load original sQTL sequences

        Returns:
            List of sQTLSample objects with original DNA sequences
        """
        print("\nLoading sQTL data from genomic-FM data wrapper...")

        # Use genomic-FM data wrapper to get original sequences
        data_loader = sQTLDataWrapper(
            num_records=self.num_samples,
            all_records=self.all_records
        )
        raw_data = data_loader.get_data(Seq_length=self.seq_length)

        print(f"Loaded {len(raw_data)} raw records")

        # Parse into sQTLSample objects
        samples = []
        for i, record in enumerate(raw_data):
            # Record format: [[ref_seq, alt_seq, tissue], label]
            x, y = record
            ref_seq, alt_seq, tissue = x

            # Map label
            if isinstance(y, str):
                label = self.label_map.get(y, -1)
                label_name = y
            else:
                label = y
                label_name = self.label_names.get(y, "unknown")

            # Filter by tissue if specified
            if self.tissue_filter and tissue != self.tissue_filter:
                continue

            # Find variant position (where sequences differ)
            variant_pos = self._find_variant_position(ref_seq, alt_seq)

            sample = sQTLSample(
                ref_sequence=ref_seq,
                alt_sequence=alt_seq,
                tissue=tissue,
                label=label,
                label_name=label_name,
                variant_position=variant_pos
            )
            samples.append(sample)

        print(f"\nProcessed {len(samples)} samples")
        print(f"  Significant: {sum(s.label == 0 for s in samples)}")
        print(f"  Not significant: {sum(s.label == 1 for s in samples)}")

        # Print tissue distribution
        tissues = {}
        for s in samples:
            tissues[s.tissue] = tissues.get(s.tissue, 0) + 1
        print(f"  Tissues: {tissues}")

        # Store samples for later use
        self.samples = samples

        return samples

    def _find_variant_position(self, ref_seq: str, alt_seq: str) -> int:
        """Find the position where reference and alternate sequences differ"""
        if len(ref_seq) != len(alt_seq):
            return -1

        differences = [i for i in range(len(ref_seq)) if ref_seq[i] != alt_seq[i]]

        if not differences:
            return -1
        elif len(differences) == 1:
            return differences[0]
        else:
            # Multiple differences - return middle position
            return differences[len(differences) // 2]

    def get_label_name(self, label: int) -> str:
        """Get human-readable label name"""
        return self.label_names.get(label, "unknown")

    def get_statistics(self, samples: List[sQTLSample]) -> Dict:
        """Get dataset statistics"""
        stats = {
            'total_samples': len(samples),
            'sequence_length': self.seq_length,
            'label_distribution': {},
            'tissue_distribution': {},
            'variant_positions': []
        }

        for sample in samples:
            # Label distribution
            stats['label_distribution'][sample.label_name] = \
                stats['label_distribution'].get(sample.label_name, 0) + 1

            # Tissue distribution
            stats['tissue_distribution'][sample.tissue] = \
                stats['tissue_distribution'].get(sample.tissue, 0) + 1

            # Variant positions
            if sample.variant_position >= 0:
                stats['variant_positions'].append(sample.variant_position)

        # Calculate class balance
        if len(stats['label_distribution']) == 2:
            counts = list(stats['label_distribution'].values())
            stats['class_balance_ratio'] = max(counts) / min(counts)

        # Variant position statistics
        if stats['variant_positions']:
            stats['variant_position_mean'] = np.mean(stats['variant_positions'])
            stats['variant_position_std'] = np.std(stats['variant_positions'])
            stats['variant_position_median'] = np.median(stats['variant_positions'])

        return stats

    def print_statistics(self, samples: List[sQTLSample] = None):
        """Print formatted statistics"""
        if samples is None:
            samples = self.samples
        stats = self.get_statistics(samples)

        print("\n" + "="*60)
        print("sQTL Dataset Statistics")
        print("="*60)
        print(f"Total samples: {stats['total_samples']}")
        print(f"Sequence length: {stats['sequence_length']}bp")

        print("\nLabel Distribution:")
        for label, count in stats['label_distribution'].items():
            pct = 100 * count / stats['total_samples']
            print(f"  {label}: {count} ({pct:.1f}%)")

        if 'class_balance_ratio' in stats:
            print(f"  Imbalance ratio: {stats['class_balance_ratio']:.2f}:1")

        print("\nTissue Distribution:")
        for tissue, count in stats['tissue_distribution'].items():
            pct = 100 * count / stats['total_samples']
            print(f"  {tissue}: {count} ({pct:.1f}%)")

        if stats['variant_positions']:
            print("\nVariant Position Statistics:")
            print(f"  Mean: {stats['variant_position_mean']:.1f}")
            print(f"  Median: {stats['variant_position_median']:.1f}")
            print(f"  Std: {stats['variant_position_std']:.1f}")

        print("="*60)

    def create_mini_batch(self,
                         samples: List[sQTLSample],
                         start_idx: int,
                         batch_size: int) -> Tuple[List[str], List[str], np.ndarray, List[str]]:
        """
        Create a mini-batch for model input

        Args:
            samples: List of sQTLSample objects
            start_idx: Starting index
            batch_size: Batch size

        Returns:
            (ref_sequences, alt_sequences, labels, tissues)
        """
        end_idx = min(start_idx + batch_size, len(samples))
        batch = samples[start_idx:end_idx]

        ref_seqs = [s.ref_sequence for s in batch]
        alt_seqs = [s.alt_sequence for s in batch]
        labels = np.array([s.label for s in batch])
        tissues = [s.tissue for s in batch]

        return ref_seqs, alt_seqs, labels, tissues


def demo_usage():
    """Demonstrate how to use the data loader"""
    print("="*60)
    print("sQTL Original Data Loader Demo")
    print("="*60)

    # Load small sample
    loader = OriginalSQTLDataLoader(
        num_samples=100,
        tissue_filter="Whole_Blood"
    )

    samples = loader.load_data()
    loader.print_statistics(samples)

    # Show example samples
    print("\nExample Samples:")
    for i in range(min(3, len(samples))):
        sample = samples[i]
        print(f"\nSample {i+1}:")
        print(f"  {sample}")
        print(f"  Ref (first 60bp): {sample.ref_sequence[:60]}...")
        print(f"  Alt (first 60bp): {sample.alt_sequence[:60]}...")
        if sample.variant_position >= 0:
            pos = sample.variant_position
            print(f"  Variant at position {pos}:")
            print(f"    Ref: {sample.ref_sequence[max(0,pos-5):pos+6]}")
            print(f"    Alt: {sample.alt_sequence[max(0,pos-5):pos+6]}")

    return samples


# Alias for compatibility with generic data loader
sQTLDataLoader = OriginalSQTLDataLoader


if __name__ == "__main__":
    samples = demo_usage()
