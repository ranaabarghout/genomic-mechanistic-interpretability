"""Run complete interpretability analysis pipeline."""
import os
import sys
import argparse
import torch
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.load_model import load_base_model, load_finetuned_model, get_model_info


def run_full_analysis(
    model_path: str,
    checkpoint_path: str = None,
    data_dir: str = None,
    output_dir: str = './full_analysis',
    sequence: str = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    run_data_exploration: bool = True,
    run_attention: bool = True,
    run_ablation: bool = True,
    run_sae: bool = False  # SAE is expensive, off by default
):
    """Run complete interpretability analysis pipeline.

    Args:
        model_path: Path to base model
        checkpoint_path: Optional path to fine-tuned checkpoint
        data_dir: Directory containing training data
        output_dir: Directory for all outputs
        sequence: Optional DNA sequence to analyze
        device: Device to run on
        run_data_exploration: Whether to run data exploration
        run_attention: Whether to run attention analysis
        run_ablation: Whether to run ablation studies
        run_sae: Whether to train sparse autoencoder
    """
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, f"analysis_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print("GENOMIC MODEL INTERPRETABILITY ANALYSIS")
    print(f"{'='*80}")
    print(f"Timestamp: {timestamp}")
    print(f"Output directory: {output_dir}")
    print(f"Model: {model_path}")
    if checkpoint_path:
        print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")

    # === 1. Model Loading and Inspection ===
    print("\n" + "="*80)
    print("STEP 1: MODEL LOADING")
    print("="*80 + "\n")

    if checkpoint_path:
        print(f"Loading fine-tuned model from {checkpoint_path}...")
        model, tokenizer, config = load_finetuned_model(
            checkpoint_path, model_path, device=device
        )
    else:
        print(f"Loading base model from {model_path}...")
        model, tokenizer, config = load_base_model(model_path, device=device)

    print("\nModel architecture:")
    get_model_info(model)

    # === 2. Data Exploration ===
    if run_data_exploration and data_dir:
        print("\n" + "="*80)
        print("STEP 2: DATA EXPLORATION")
        print("="*80 + "\n")

        data_output_dir = os.path.join(output_dir, 'data_exploration')
        os.makedirs(data_output_dir, exist_ok=True)

        try:
            from data.explore_data import create_data_exploration_report

            print(f"Exploring data in {data_dir}...")
            create_data_exploration_report(data_dir, data_output_dir)
            print(f"Data exploration complete. Results in {data_output_dir}")
        except Exception as e:
            print(f"Error in data exploration: {e}")
            print("Continuing with other analyses...")

    # === 3. Attention Analysis ===
    if run_attention:
        print("\n" + "="*80)
        print("STEP 3: ATTENTION ANALYSIS")
        print("="*80 + "\n")

        attention_output_dir = os.path.join(output_dir, 'attention')
        os.makedirs(attention_output_dir, exist_ok=True)

        try:
            # Import here to avoid loading if not needed
            from scripts.run_attention_analysis import run_attention_analysis

            run_attention_analysis(
                model_path=model_path,
                checkpoint_path=checkpoint_path,
                sequence=sequence,
                output_dir=attention_output_dir,
                device=device
            )
        except Exception as e:
            print(f"Error in attention analysis: {e}")
            print("Continuing with other analyses...")

    # === 4. Ablation Study ===
    if run_ablation:
        print("\n" + "="*80)
        print("STEP 4: ABLATION STUDY")
        print("="*80 + "\n")

        ablation_output_dir = os.path.join(output_dir, 'ablation')
        os.makedirs(ablation_output_dir, exist_ok=True)

        try:
            from scripts.run_ablation_study import run_ablation_study

            run_ablation_study(
                model_path=model_path,
                checkpoint_path=checkpoint_path,
                sequence=sequence,
                output_dir=ablation_output_dir,
                device=device
            )
        except Exception as e:
            print(f"Error in ablation study: {e}")
            print("Continuing with other analyses...")

    # === 5. Sparse Autoencoder (Optional) ===
    if run_sae:
        print("\n" + "="*80)
        print("STEP 5: SPARSE AUTOENCODER")
        print("="*80 + "\n")

        sae_output_dir = os.path.join(output_dir, 'sae')
        os.makedirs(sae_output_dir, exist_ok=True)

        try:
            print("Training sparse autoencoder (this may take a while)...")
            print("Note: Not yet implemented - placeholder for future work")
            # TODO: Implement SAE training pipeline
        except Exception as e:
            print(f"Error in SAE training: {e}")

    # === Final Summary ===
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}")
    print("\nGenerated outputs:")
    if run_data_exploration and data_dir:
        print(f"  - Data exploration: {os.path.join(output_dir, 'data_exploration')}")
    if run_attention:
        print(f"  - Attention analysis: {os.path.join(output_dir, 'attention')}")
    if run_ablation:
        print(f"  - Ablation study: {os.path.join(output_dir, 'ablation')}")
    if run_sae:
        print(f"  - Sparse autoencoder: {os.path.join(output_dir, 'sae')}")

    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run complete interpretability analysis on genomic model'
    )

    parser.add_argument(
        '--model_path',
        type=str,
        default='/project/def-mahadeva/ranaab/genomic-FM/models/dnabert2',
        help='Path to base model'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to fine-tuned checkpoint (.ckpt file)'
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default='/project/def-mahadeva/ranaab/genomic-FM/root/data',
        help='Directory containing training data'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./full_analysis',
        help='Base output directory'
    )

    parser.add_argument(
        '--sequence',
        type=str,
        default=None,
        help='DNA sequence to analyze'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run on (cuda/cpu)'
    )

    parser.add_argument(
        '--skip-data',
        action='store_true',
        help='Skip data exploration'
    )

    parser.add_argument(
        '--skip-attention',
        action='store_true',
        help='Skip attention analysis'
    )

    parser.add_argument(
        '--skip-ablation',
        action='store_true',
        help='Skip ablation study'
    )

    parser.add_argument(
        '--run-sae',
        action='store_true',
        help='Run sparse autoencoder training (expensive)'
    )

    args = parser.parse_args()

    run_full_analysis(
        model_path=args.model_path,
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        sequence=args.sequence,
        device=args.device,
        run_data_exploration=not args.skip_data,
        run_attention=not args.skip_attention,
        run_ablation=not args.skip_ablation,
        run_sae=args.run_sae
    )


if __name__ == "__main__":
    main()
