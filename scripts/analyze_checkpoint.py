"""Quick script to run analysis on the trained checkpoint."""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from scripts.run_full_analysis import run_full_analysis

# Paths from your Trillium cluster setup
MODEL_PATH = 'zhihan1996/DNABERT-2-117M'  # Using HuggingFace name
CHECKPOINT_PATH = '/project/def-mahadeva/ranaab/genomic-FM/GV-Rep/9p1e0e6n/checkpoints/epoch=99-step=18800.ckpt'
DATA_DIR = '/project/def-mahadeva/ranaab/genomic-FM/root/data'
OUTPUT_DIR = './checkpoint_analysis'

# Example sequences for analysis
EXAMPLE_SEQUENCES = {
    'wildtype': 'ATCGATCGATCGATCGATCGATCGATCGATCG' * 4,  # 128bp wildtype
    'variant': 'ATCGATCGATCGATCGTTTGATCGATCGATCG' * 4,   # 128bp with variant
}


def main():
    print("="*80)
    print("ANALYZING TRAINED CHECKPOINT")
    print("="*80)
    print(f"\nModel: {MODEL_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Data: {DATA_DIR}")
    print(f"Output: {OUTPUT_DIR}\n")

    # Check if checkpoint exists
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\n⚠️  WARNING: Checkpoint not found at {CHECKPOINT_PATH}")
        print("You may need to update CHECKPOINT_PATH in this script.")
        print("If running from login node, the checkpoint might be on /scratch.")
        print("\nProceeding with base model analysis only...\n")
        checkpoint = None
    else:
        checkpoint = CHECKPOINT_PATH
        print(f"✓ Checkpoint found\n")

    # Run full analysis
    print("Starting comprehensive analysis...\n")

    try:
        run_full_analysis(
            model_path=MODEL_PATH,
            checkpoint_path=checkpoint,
            data_dir=DATA_DIR,
            output_dir=OUTPUT_DIR,
            sequence=EXAMPLE_SEQUENCES['wildtype'],  # Use wildtype for main analysis
            device='cuda',
            run_data_exploration=True,
            run_attention=True,
            run_ablation=True,
            run_sae=False  # Set to True if you want SAE (takes longer)
        )

        print("\n" + "="*80)
        print("✓ ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nResults saved to: {OUTPUT_DIR}")
        print("\nTo compare wildtype vs variant:")
        print("  1. Run attention analysis on wildtype (done above)")
        print("  2. Run attention analysis on variant:")
        print(f"     python scripts/run_attention_analysis.py \\")
        print(f"       --model_path {MODEL_PATH} \\")
        print(f"       --checkpoint {CHECKPOINT_PATH} \\")
        print(f"       --sequence '{EXAMPLE_SEQUENCES['variant']}' \\")
        print(f"       --output_dir ./variant_attention")

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
