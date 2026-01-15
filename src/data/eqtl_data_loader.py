"""
eQTL (Expression QTL) data loader for mechanistic interpretability analysis.
Very similar to sQTL - analyzes variants affecting gene expression levels.
"""
from pathlib import Path
from typing import List, Optional
import sys
from tqdm import tqdm

# Add genomic-FM to path before any imports from it
# Need to resolve() to handle relative paths like scripts/../src properly
GENOMIC_FM_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "genomic-FM"
GENOMIC_FM_SRC = GENOMIC_FM_ROOT / "src"
for path in [str(GENOMIC_FM_ROOT), str(GENOMIC_FM_SRC)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from data.generic_data_loader import GenericDataLoader, GenomicSample
from src.dataloader.data_wrapper import eQTLDataWrapper


class eQTLDataLoader(GenericDataLoader):
    """
    Data loader for eQTL (Expression Quantitative Trait Loci) variants.

    eQTLs are genetic variants that affect gene expression levels.
    Similar to sQTLs but affect overall expression rather than splicing.
    """

    def __init__(
        self,
        num_records: int = 1000,
        seq_length: int = 1024,
        all_records: bool = False,
        tissue_filter: Optional[str] = None
    ):
        """
        Initialize eQTL data loader.

        Args:
            num_records: Number of records to load
            seq_length: Length of sequences (bp)
            all_records: Load all available records
            tissue_filter: Filter to specific tissue (e.g., 'Whole_Blood')
        """
        super().__init__(
            dataset_name='eqtl',
            num_records=num_records,
            seq_length=seq_length,
            all_records=all_records,
            filter_criteria={'tissue': tissue_filter}
        )

        self.tissue_filter = tissue_filter
        self.wrapper = None

    def load_data(self) -> List[GenomicSample]:
        """Load eQTL data from genomic-FM wrapper."""
        print(f"Initializing eQTL data loader...")
        print(f"  Sequence length: {self.seq_length}bp")
        if self.tissue_filter:
            print(f"  Tissue filter: {self.tissue_filter}")

        print("\nLoading eQTL data from genomic-FM data wrapper...")
        self.wrapper = eQTLDataWrapper(
            num_records=self.num_records,
            all_records=self.all_records
        )

        # Get data
        self.raw_data = self.wrapper.get_data(Seq_length=self.seq_length)
        print(f"Loaded {len(self.raw_data)} raw records")

        # Parse into GenomicSample objects
        self.samples = []
        for record in tqdm(self.raw_data, desc="Processing"):
            sample = self.parse_sample(record)
            if sample is not None:
                # Apply tissue filter if specified
                if self.tissue_filter is None or sample.metadata.get('tissue') == self.tissue_filter:
                    self.samples.append(sample)

        print(f"\nProcessed {len(self.samples)} samples")

        # Print label distribution
        label_counts = self.get_label_counts()
        for label, count in label_counts.items():
            print(f"  {label}: {count}")

        # Print tissue distribution
        tissues = {}
        for sample in self.samples:
            tissue = sample.metadata.get('tissue', 'Unknown')
            tissues[tissue] = tissues.get(tissue, 0) + 1
        print(f"  Tissues: {tissues}")

        return self.samples

    def parse_sample(self, record) -> Optional[GenomicSample]:
        """Parse eQTL record into GenomicSample."""
        try:
            # Record format from genomic-FM: [[ref_seq, alt_seq, tissue], label]
            if isinstance(record, (list, tuple)) and len(record) == 2:
                x, y = record

                # Extract sequences and metadata
                if isinstance(x, (list, tuple)) and len(x) >= 2:
                    ref_seq = x[0]
                    alt_seq = x[1]
                    tissue = x[2] if len(x) > 2 else 'Unknown'
                else:
                    return None
            else:
                return None

            if not ref_seq:
                return None

            # Parse label - eQTL has binary (significant/not) or continuous
            # Convert label (0=significant, 1=not_significant to match sQTL convention)
            if isinstance(y, str):
                label = 1 if y.lower() in ['negative', 'not_significant', 'no', 'false', '0'] else 0
            else:
                label = int(y)

            # Find variant position (where sequences differ)
            variant_pos = self._find_variant_position(ref_seq, alt_seq)

            # Get label name
            label_name = self.get_label_name(label)

            # Extract metadata
            metadata = {
                'tissue': tissue,
                'gene': None,  # Not available in basic format
                'pvalue': None,
                'effect_size': None,
                'chromosome': None,
                'position': None
            }

            return GenomicSample(
                ref_sequence=ref_seq,
                alt_sequence=alt_seq,
                label=label,
                label_name=label_name,
                variant_pos=variant_pos,
                metadata=metadata
            )

        except Exception as e:
            print(f"Warning: Failed to parse record: {e}")
            return None

    def _find_variant_position(self, ref_seq: str, alt_seq: str) -> int:
        """Find position where sequences differ."""
        if not alt_seq or len(ref_seq) != len(alt_seq):
            return len(ref_seq) // 2

        for i, (r, a) in enumerate(zip(ref_seq, alt_seq)):
            if r != a:
                return i

        return len(ref_seq) // 2

    def get_label_name(self, label: int) -> str:
        """Get human-readable label name."""
        return 'significant' if label == 0 else 'not_significant'

    @property
    def significant(self) -> List[GenomicSample]:
        """Get all significant eQTL samples."""
        return self.filter_by_label(0)

    @property
    def not_significant(self) -> List[GenomicSample]:
        """Get all not significant eQTL samples."""
        return self.filter_by_label(1)


if __name__ == "__main__":
    """Test eQTL data loader."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    print("Testing eQTL Data Loader")
    print("="*60)

    loader = eQTLDataLoader(num_records=100, seq_length=1024)
    samples = loader.load_data()

    loader.print_statistics()

    print(f"\nExample samples:")
    for i, sample in enumerate(samples[:3]):
        print(f"{i+1}. {sample}")
