"""
MAVE (Multiplexed Assays of Variant Effects) data loader.
Typically continuous regression task for variant effect prediction.
"""
from typing import List, Optional
from tqdm import tqdm
import numpy as np
from data.generic_data_loader import GenericDataLoader, GenomicSample, GENOMIC_FM_PATH

import sys
if str(GENOMIC_FM_PATH) not in sys.path:
    sys.path.insert(0, str(GENOMIC_FM_PATH))

from src.dataloader.data_wrapper import MAVEDataWrapper


class MAVEDataLoader(GenericDataLoader):
    """
    Data loader for MAVE (Multiplexed Assays of Variant Effects).

    MAVE provides experimental measurements of variant effects,
    typically as continuous scores. Can be used for:
    - Regression: predict continuous effect score
    - Classification: bin scores into categories (e.g., gain/loss/neutral)
    """

    def __init__(
        self,
        num_records: int = 1000,
        seq_length: int = 1024,
        all_records: bool = False,
        classification: bool = False,
        score_thresholds: Optional[tuple] = None
    ):
        """
        Initialize MAVE data loader.

        Args:
            num_records: Number of records to load
            seq_length: Length of sequences (bp)
            all_records: Load all available records
            classification: Convert continuous scores to categories
            score_thresholds: Thresholds for binning (low, high) if classification=True
                             Default: (-0.5, 0.5) for loss-of-function, neutral, gain-of-function
        """
        super().__init__(
            dataset_name='mave',
            num_records=num_records,
            seq_length=seq_length,
            all_records=all_records
        )

        self.classification = classification
        self.score_thresholds = score_thresholds or (-0.5, 0.5)
        self.wrapper = None

    def load_data(self) -> List[GenomicSample]:
        """Load MAVE data from genomic-FM wrapper."""
        print(f"Initializing MAVE data loader...")
        print(f"  Sequence length: {self.seq_length}bp")
        print(f"  Classification mode: {self.classification}")
        if self.classification:
            print(f"  Score thresholds: {self.score_thresholds}")

        print("\nLoading MAVE data from genomic-FM data wrapper...")
        self.wrapper = MAVEDataWrapper(
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
                self.samples.append(sample)

        print(f"\nProcessed {len(self.samples)} samples")

        # Print label/score statistics
        if self.classification:
            label_counts = self.get_label_counts()
            for label, count in label_counts.items():
                print(f"  {label}: {count}")
        else:
            scores = [s.label for s in self.samples]
            print(f"  Score range: [{min(scores):.3f}, {max(scores):.3f}]")
            print(f"  Score mean: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

        return self.samples

    def parse_sample(self, record) -> Optional[GenomicSample]:
        """Parse MAVE record into GenomicSample."""
        try:
            # Extract sequences
            ref_seq = record.get('ref_seq', record.get('sequence_ref', record.get('wild_type', '')))
            alt_seq = record.get('alt_seq', record.get('sequence_alt', record.get('mutant', '')))

            if not ref_seq:
                return None

            # Get continuous score
            score = record.get('score', record.get('effect_score', record.get('fitness', None)))

            if score is None:
                return None

            score = float(score)

            # Convert to classification if requested
            if self.classification:
                low_thresh, high_thresh = self.score_thresholds
                if score < low_thresh:
                    label = 0  # Loss of function
                elif score > high_thresh:
                    label = 2  # Gain of function
                else:
                    label = 1  # Neutral
            else:
                label = score  # Keep continuous for regression

            # Get variant position
            variant_pos = record.get('variant_position', record.get('position', len(ref_seq) // 2))

            # Extract metadata
            metadata = {
                'score': score,  # Always keep original score
                'gene': record.get('gene', record.get('target', None)),
                'assay': record.get('assay', record.get('experiment', None)),
                'variant_type': record.get('variant_type', record.get('mutation_type', None)),
                'amino_acid_change': record.get('aa_change', None),
                'standard_error': record.get('se', record.get('standard_error', None))
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

    def get_label_name(self, label) -> str:
        """Get human-readable label name."""
        if self.classification:
            if label == 0:
                return 'loss_of_function'
            elif label == 1:
                return 'neutral'
            elif label == 2:
                return 'gain_of_function'
            else:
                return 'unknown'
        else:
            return f'score_{label:.3f}'

    @property
    def loss_of_function(self) -> List[GenomicSample]:
        """Get loss of function variants (only in classification mode)."""
        if not self.classification:
            return []
        return self.filter_by_label(0)

    @property
    def neutral(self) -> List[GenomicSample]:
        """Get neutral variants (only in classification mode)."""
        if not self.classification:
            return []
        return self.filter_by_label(1)

    @property
    def gain_of_function(self) -> List[GenomicSample]:
        """Get gain of function variants (only in classification mode)."""
        if not self.classification:
            return []
        return self.filter_by_label(2)


if __name__ == "__main__":
    """Test MAVE data loader."""
    print("Testing MAVE Data Loader (Classification Mode)")
    print("="*60)

    loader = MAVEDataLoader(
        num_records=100,
        seq_length=1024,
        classification=True,
        score_thresholds=(-0.5, 0.5)
    )
    samples = loader.load_data()

    loader.print_statistics()

    print(f"\nExample samples:")
    for i, sample in enumerate(samples[:3]):
        print(f"{i+1}. {sample}")

    print("\n" + "="*60)
    print("Testing MAVE Data Loader (Regression Mode)")
    print("="*60)

    loader2 = MAVEDataLoader(
        num_records=100,
        seq_length=1024,
        classification=False
    )
    samples2 = loader2.load_data()

    print(f"\nExample samples:")
    for i, sample in enumerate(samples2[:3]):
        print(f"{i+1}. {sample}")
