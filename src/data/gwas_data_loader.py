"""
GWAS (Genome-Wide Association Study) data loader for mechanistic interpretability.
Analyzes trait-associated genetic variants.
"""
from typing import List, Optional
from tqdm import tqdm
from data.generic_data_loader import GenericDataLoader, GenomicSample, GENOMIC_FM_PATH

import sys
if str(GENOMIC_FM_PATH) not in sys.path:
    sys.path.insert(0, str(GENOMIC_FM_PATH))

from src.dataloader.data_wrapper import GWASDataWrapper


class GWASDataLoader(GenericDataLoader):
    """
    Data loader for GWAS (Genome-Wide Association Study) variants.

    GWAS variants are associated with complex traits and diseases.
    Can be used for:
    - Binary: associated vs not associated with trait
    - Regression: effect size or p-value prediction
    """

    def __init__(
        self,
        num_records: int = 1000,
        seq_length: int = 1024,
        all_records: bool = False,
        trait_filter: Optional[str] = None,
        pvalue_threshold: float = 5e-8
    ):
        """
        Initialize GWAS data loader.

        Args:
            num_records: Number of records to load
            seq_length: Length of sequences (bp)
            all_records: Load all available records
            trait_filter: Filter to specific trait/phenotype
            pvalue_threshold: P-value threshold for significance (default: 5e-8)
        """
        super().__init__(
            dataset_name='gwas',
            num_records=num_records,
            seq_length=seq_length,
            all_records=all_records,
            filter_criteria={'trait': trait_filter}
        )

        self.trait_filter = trait_filter
        self.pvalue_threshold = pvalue_threshold
        self.wrapper = None

    def load_data(self) -> List[GenomicSample]:
        """Load GWAS data from genomic-FM wrapper."""
        print(f"Initializing GWAS data loader...")
        print(f"  Sequence length: {self.seq_length}bp")
        print(f"  P-value threshold: {self.pvalue_threshold}")
        if self.trait_filter:
            print(f"  Trait filter: {self.trait_filter}")

        print("\nLoading GWAS data from genomic-FM data wrapper...")
        self.wrapper = GWASDataWrapper(
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
                # Apply trait filter if specified
                if self.trait_filter is None or sample.metadata.get('trait') == self.trait_filter:
                    self.samples.append(sample)

        print(f"\nProcessed {len(self.samples)} samples")

        # Print label distribution
        label_counts = self.get_label_counts()
        for label, count in label_counts.items():
            print(f"  {label}: {count}")

        # Print trait distribution
        traits = {}
        for sample in self.samples:
            trait = sample.metadata.get('trait', 'Unknown')
            traits[trait] = traits.get(trait, 0) + 1
        if len(traits) < 20:
            print(f"  Traits: {traits}")
        else:
            print(f"  Unique traits: {len(traits)}")

        return self.samples

    def parse_sample(self, record) -> Optional[GenomicSample]:
        """Parse GWAS record into GenomicSample."""
        try:
            # Extract sequences
            ref_seq = record.get('ref_seq', record.get('sequence_ref', ''))
            alt_seq = record.get('alt_seq', record.get('sequence_alt', ''))

            if not ref_seq:
                return None

            # Parse label based on p-value or explicit label
            pvalue = record.get('pvalue', record.get('p_value', record.get('p', None)))

            if pvalue is not None:
                # Binary: significant vs not significant
                label = 1 if float(pvalue) < self.pvalue_threshold else 0
            else:
                # Use explicit label if available
                label_raw = record.get('label', record.get('significant', 0))
                if isinstance(label_raw, str):
                    label = 1 if label_raw.lower() in ['significant', 'associated', 'yes', 'true', '1'] else 0
                else:
                    label = int(label_raw)

            # Get variant position
            variant_pos = record.get('variant_position', len(ref_seq) // 2)

            # Extract metadata
            metadata = {
                'trait': record.get('trait', record.get('phenotype', 'Unknown')),
                'pvalue': pvalue,
                'effect_size': record.get('effect_size', record.get('beta', record.get('odds_ratio', None))),
                'chromosome': record.get('chromosome', record.get('chr', None)),
                'position': record.get('position', record.get('pos', None)),
                'rsid': record.get('rsid', record.get('snp_id', None)),
                'allele_freq': record.get('allele_frequency', record.get('maf', None))
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
        return 'significant' if label == 1 else 'not_significant'

    @property
    def significant(self) -> List[GenomicSample]:
        """Get all trait-associated variant samples."""
        return self.filter_by_label(1)

    @property
    def not_significant(self) -> List[GenomicSample]:
        """Get all non-associated variant samples."""
        return self.filter_by_label(0)


if __name__ == "__main__":
    """Test GWAS data loader."""
    print("Testing GWAS Data Loader")
    print("="*60)

    loader = GWASDataLoader(num_records=100, seq_length=1024)
    samples = loader.load_data()

    loader.print_statistics()

    print(f"\nExample samples:")
    for i, sample in enumerate(samples[:3]):
        print(f"{i+1}. {sample}")
