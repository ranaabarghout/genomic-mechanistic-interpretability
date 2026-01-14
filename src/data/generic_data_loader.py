"""
Generic data loader base class for genomic variant datasets.
Works with any dataset from genomic-FM repository.
"""
import sys
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import numpy as np

# Add genomic-FM to path - need both root (for package) and src (for direct imports)
GENOMIC_FM_ROOT = Path(__file__).parent.parent.parent.parent / "genomic-FM"
GENOMIC_FM_PATH = GENOMIC_FM_ROOT / "src"
for path in [str(GENOMIC_FM_ROOT), str(GENOMIC_FM_PATH)]:
    if path not in sys.path:
        sys.path.insert(0, path)


@dataclass
class GenomicSample:
    """Generic container for genomic variant samples."""
    ref_sequence: str
    alt_sequence: Optional[str] = None
    label: Union[str, int, float] = None
    variant_pos: int = 512  # Default center position
    metadata: Dict = None
    label_name: str = None  # Human-readable label name

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def seq_len(self):
        return len(self.ref_sequence)

    def __repr__(self):
        meta_str = ", ".join(f"{k}={v}" for k, v in self.metadata.items() if v is not None)
        return f"GenomicSample(label={self.label}, variant_pos={self.variant_pos}, seq_len={self.seq_len}, {meta_str})"


class GenericDataLoader:
    """
    Generic data loader for any genomic variant dataset.

    This base class provides common functionality for loading and processing
    genomic variant data from the genomic-FM repository.
    """

    def __init__(
        self,
        dataset_name: str,
        num_records: int = 1000,
        seq_length: int = 1024,
        all_records: bool = False,
        filter_criteria: Optional[Dict] = None
    ):
        """
        Initialize generic data loader.

        Args:
            dataset_name: Name of dataset (e.g., 'eqtl', 'clinvar', 'gwas')
            num_records: Number of records to load (if not all_records)
            seq_length: Length of sequences to extract
            all_records: Whether to load all available records
            filter_criteria: Optional filtering criteria (e.g., tissue, chromosome)
        """
        self.dataset_name = dataset_name
        self.num_records = num_records
        self.seq_length = seq_length
        self.all_records = all_records
        self.filter_criteria = filter_criteria or {}

        self.samples: List[GenomicSample] = []
        self.raw_data = None
        self.label_distribution = {}

    def load_data(self) -> List[GenomicSample]:
        """
        Load data from genomic-FM wrapper.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement load_data()")

    def parse_sample(self, record) -> Optional[GenomicSample]:
        """
        Parse a single record into GenomicSample.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement parse_sample()")

    def get_label_name(self, label: Union[str, int, float]) -> str:
        """
        Get human-readable label name.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement get_label_name()")

    def filter_by_label(self, label: Union[str, int, float]) -> List[GenomicSample]:
        """Filter samples by label."""
        return [s for s in self.samples if s.label == label]

    def get_label_counts(self) -> Dict[str, int]:
        """Get count of samples per label."""
        counts = {}
        for sample in self.samples:
            label_name = self.get_label_name(sample.label)
            counts[label_name] = counts.get(label_name, 0) + 1
        return counts

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        if not self.samples:
            return {}

        stats = {
            'total_samples': len(self.samples),
            'sequence_length': self.seq_length,
            'label_distribution': self.get_label_counts(),
            'variant_positions': {
                'mean': np.mean([s.variant_pos for s in self.samples]),
                'median': np.median([s.variant_pos for s in self.samples]),
                'std': np.std([s.variant_pos for s in self.samples])
            }
        }

        # Add metadata statistics if available
        if self.samples[0].metadata:
            for key in self.samples[0].metadata.keys():
                values = [s.metadata.get(key) for s in self.samples if s.metadata.get(key)]
                if values and isinstance(values[0], str):
                    unique = list(set(values))
                    if len(unique) < 20:  # Only track if not too many unique values
                        stats[f'{key}_distribution'] = {v: values.count(v) for v in unique}

        return stats

    def print_statistics(self):
        """Print formatted dataset statistics."""
        stats = self.get_statistics()

        print("\n" + "="*60)
        print(f"{self.dataset_name.upper()} Dataset Statistics")
        print("="*60)

        if not stats or 'total_samples' not in stats:
            print("No samples loaded!")
            print("="*60 + "\n")
            return

        print(f"Total samples: {stats['total_samples']}")
        print(f"Sequence length: {stats['sequence_length']}bp")

        print("\nLabel Distribution:")
        for label, count in stats['label_distribution'].items():
            pct = 100 * count / stats['total_samples']
            print(f"  {label}: {count} ({pct:.1f}%)")

        # Calculate imbalance ratio
        counts = list(stats['label_distribution'].values())
        if len(counts) > 1:
            imbalance = max(counts) / min(counts)
            print(f"  Imbalance ratio: {imbalance:.2f}:1")

        print("\nVariant Position Statistics:")
        print(f"  Mean: {stats['variant_positions']['mean']:.1f}")
        print(f"  Median: {stats['variant_positions']['median']:.1f}")
        print(f"  Std: {stats['variant_positions']['std']:.1f}")

        # Print metadata distributions
        for key, dist in stats.items():
            if key.endswith('_distribution') and key != 'label_distribution':
                print(f"\n{key.replace('_', ' ').title()}:")
                for value, count in sorted(dist.items(), key=lambda x: x[1], reverse=True)[:5]:
                    pct = 100 * count / stats['total_samples']
                    print(f"  {value}: {count} ({pct:.1f}%)")

        print("="*60 + "\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def get_data_loader(dataset_name: str, **kwargs) -> GenericDataLoader:
    """
    Factory function to get appropriate data loader for dataset.

    Args:
        dataset_name: Name of dataset ('eqtl', 'sqtl', 'clinvar', 'gwas', 'mave', etc.)
        **kwargs: Additional arguments passed to loader

    Returns:
        Appropriate data loader instance
    """
    dataset_name = dataset_name.lower()

    if dataset_name == 'sqtl':
        from data.sqtl_data_loader import sQTLDataLoader
        return sQTLDataLoader(**kwargs)
    elif dataset_name == 'eqtl':
        from data.eqtl_data_loader import eQTLDataLoader
        return eQTLDataLoader(**kwargs)
    elif dataset_name == 'clinvar':
        from data.clinvar_data_loader import ClinVarDataLoader
        return ClinVarDataLoader(**kwargs)
    elif dataset_name == 'gwas':
        from data.gwas_data_loader import GWASDataLoader
        return GWASDataLoader(**kwargs)
    elif dataset_name == 'mave':
        from data.mave_data_loader import MAVEDataLoader
        return MAVEDataLoader(**kwargs)
    elif dataset_name == 'geneko':
        from data.geneko_data_loader import GeneKoDataLoader
        return GeneKoDataLoader(**kwargs)
    elif dataset_name == 'cellpassport':
        from data.cellpassport_data_loader import CellPassportDataLoader
        return CellPassportDataLoader(**kwargs)
    elif dataset_name == 'oligogenic':
        from data.oligogenic_data_loader import OligogenicDataLoader
        return OligogenicDataLoader(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: sqtl, eqtl, clinvar, gwas, mave, geneko, cellpassport, oligogenic")
