"""
ClinVar data loader for mechanistic interpretability analysis.
Analyzes pathogenic vs benign variant classification.
"""
from typing import List, Optional
from tqdm import tqdm
from data.generic_data_loader import GenericDataLoader, GenomicSample, GENOMIC_FM_PATH

import sys
if str(GENOMIC_FM_PATH) not in sys.path:
    sys.path.insert(0, str(GENOMIC_FM_PATH))

from src.dataloader.data_wrapper import ClinVarDataWrapper


class ClinVarDataLoader(GenericDataLoader):
    """
    Data loader for ClinVar pathogenicity predictions.

    ClinVar contains clinically relevant variants classified as:
    - Pathogenic
    - Likely pathogenic
    - Benign
    - Likely benign
    - Uncertain significance (VUS)
    """

    # Label mapping
    LABEL_MAP = {
        'pathogenic': 0,
        'likely_pathogenic': 1,
        'benign': 2,
        'likely_benign': 3,
        'uncertain': 4,
        'vus': 4,  # Variant of Uncertain Significance
    }

    LABEL_NAMES = {
        0: 'pathogenic',
        1: 'likely_pathogenic',
        2: 'benign',
        3: 'likely_benign',
        4: 'uncertain'
    }

    def __init__(
        self,
        num_records: int = 1000,
        seq_length: int = 1024,
        all_records: bool = False,
        binary_classification: bool = True,
        include_vus: bool = False
    ):
        """
        Initialize ClinVar data loader.

        Args:
            num_records: Number of records to load
            seq_length: Length of sequences (bp)
            all_records: Load all available records
            binary_classification: If True, merge into pathogenic vs benign
            include_vus: Whether to include uncertain significance variants
        """
        super().__init__(
            dataset_name='clinvar',
            num_records=num_records,
            seq_length=seq_length,
            all_records=all_records
        )

        self.binary_classification = binary_classification
        self.include_vus = include_vus
        self.wrapper = None

    def load_data(self) -> List[GenomicSample]:
        """Load ClinVar data from genomic-FM wrapper."""
        print(f"Initializing ClinVar data loader...")
        print(f"  Sequence length: {self.seq_length}bp")
        print(f"  Binary classification: {self.binary_classification}")
        print(f"  Include VUS: {self.include_vus}")

        print("\nLoading ClinVar data from genomic-FM data wrapper...")
        self.wrapper = ClinVarDataWrapper(
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
                # Filter VUS if requested
                if not self.include_vus and sample.label == 4:
                    continue
                self.samples.append(sample)

        print(f"\nProcessed {len(self.samples)} samples")

        # Print label distribution
        label_counts = self.get_label_counts()
        for label, count in label_counts.items():
            print(f"  {label}: {count}")

        return self.samples

    def parse_sample(self, record) -> Optional[GenomicSample]:
        """Parse ClinVar record into GenomicSample."""
        try:
            # Extract sequences
            ref_seq = record.get('ref_seq', record.get('sequence_ref', ''))
            alt_seq = record.get('alt_seq', record.get('sequence_alt', ''))

            if not ref_seq:
                return None

            # Parse label
            label_str = record.get('label', record.get('clinical_significance', '')).lower()

            # Map to numeric label
            label = None
            for key, value in self.LABEL_MAP.items():
                if key in label_str:
                    label = value
                    break

            if label is None:
                return None

            # Convert to binary if requested
            if self.binary_classification:
                if label in [0, 1]:  # Pathogenic or likely pathogenic
                    label = 1  # Pathogenic
                elif label in [2, 3]:  # Benign or likely benign
                    label = 0  # Benign
                else:  # VUS
                    label = 2  # Uncertain (or skip if include_vus=False)

            # Get variant position
            variant_pos = record.get('variant_position', len(ref_seq) // 2)

            # Extract metadata
            metadata = {
                'gene': record.get('gene', record.get('gene_symbol', None)),
                'variant_type': record.get('variant_type', record.get('type', None)),
                'chromosome': record.get('chromosome', record.get('chr', None)),
                'position': record.get('position', record.get('pos', None)),
                'ref_allele': record.get('ref', None),
                'alt_allele': record.get('alt', None),
                'review_status': record.get('review_status', None)
            }

            return GenomicSample(
                ref_sequence=ref_seq,
                alt_sequence=alt_seq,
                label=label,
                variant_pos=variant_pos,
                metadata=metadata
            )

        except Exception as e:
            print(f"Warning: Failed to parse record: {e}")
            return None

    def get_label_name(self, label: int) -> str:
        """Get human-readable label name."""
        if self.binary_classification:
            if label == 0:
                return 'benign'
            elif label == 1:
                return 'pathogenic'
            else:
                return 'uncertain'
        else:
            return self.LABEL_NAMES.get(label, 'unknown')

    @property
    def pathogenic(self) -> List[GenomicSample]:
        """Get all pathogenic variant samples."""
        return self.filter_by_label(1 if self.binary_classification else 0)

    @property
    def benign(self) -> List[GenomicSample]:
        """Get all benign variant samples."""
        return self.filter_by_label(0 if self.binary_classification else 2)

    @property
    def uncertain(self) -> List[GenomicSample]:
        """Get all uncertain significance variant samples."""
        return self.filter_by_label(2 if self.binary_classification else 4)


if __name__ == "__main__":
    """Test ClinVar data loader."""
    print("Testing ClinVar Data Loader")
    print("="*60)

    loader = ClinVarDataLoader(
        num_records=100,
        seq_length=1024,
        binary_classification=True,
        include_vus=False
    )
    samples = loader.load_data()

    loader.print_statistics()

    print(f"\nExample samples:")
    for i, sample in enumerate(samples[:3]):
        print(f"{i+1}. {sample}")
