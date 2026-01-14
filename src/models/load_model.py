"""Load pretrained genomic foundation model and fine-tuned checkpoints."""
import torch
from transformers import AutoTokenizer
from pathlib import Path
import sys
from typing import Optional, Tuple

def load_base_model(model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Load base DNABERT-2 model.

    Args:
        model_name: Model name or path (e.g., 'zhihan1996/DNABERT-2-117M')
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading base model: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # DNABERT-2 is a BertForMaskedLM model - load it properly
    from transformers import BertForMaskedLM
    full_model = BertForMaskedLM.from_pretrained(model_name, trust_remote_code=True)

    # Extract the base BERT model (without MLM head)
    model = full_model.bert

    # Enable attention and hidden states output
    model.config.output_attentions = True
    model.config.output_hidden_states = True

    model.to(device)
    model.eval()

    config = model.config
    print(f"Model loaded on {device}")
    print(f"Config: {config.num_hidden_layers} layers, {config.num_attention_heads} heads")

    return model, tokenizer, config


def load_finetuned_model(
    checkpoint_path: str,
    base_model_name: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """Load fine-tuned model from genomic-FM checkpoint.

    Args:
        checkpoint_path: Path to .ckpt file from fine-tuning
        base_model_name: Base model name for tokenizer
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer, config)
    """
    print(f"Loading fine-tuned checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict']

    print(f"Found {len(state_dict)} keys in checkpoint")

    # Load base model
    print(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    from transformers import BertForMaskedLM
    full_model = BertForMaskedLM.from_pretrained(base_model_name, trust_remote_code=True)
    model = full_model.bert

    # Extract BERT weights from checkpoint
    # The checkpoint has keys like 'model.bert.encoder.layer.0.attention.self.query.weight'
    bert_state_dict = {}
    for key, value in state_dict.items():
        if 'model.bert.' in key:
            # Remove 'model.' prefix
            new_key = key.replace('model.', '')
            bert_state_dict[new_key] = value
        elif 'model.backbone.' in key:
            # Handle alternative structure
            new_key = key.replace('model.backbone.', 'bert.')
            bert_state_dict[new_key] = value

    if bert_state_dict:
        print(f"Loading {len(bert_state_dict)} fine-tuned weights...")
        # Load weights (strict=False to ignore missing keys)
        missing, unexpected = model.load_state_dict(bert_state_dict, strict=False)
        print(f"✓ Fine-tuned weights loaded ({len(missing)} missing, {len(unexpected)} unexpected)")
    else:
        print("⚠ Warning: No BERT weights found in checkpoint, using base model weights")

    # Enable attention outputs
    model.config.output_attentions = True
    model.config.output_hidden_states = True

    model.to(device)
    model.eval()

    config = model.config
    print(f"Model loaded on {device}")
    print(f"Config: {config.num_hidden_layers} layers, {config.num_attention_heads} heads")

    return model, tokenizer, config


def get_model_info(model):
    """Print detailed model architecture information."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*50}")
    print(f"Model Architecture Summary")
    print(f"{'='*50}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"\nModel structure:")
    print(model)
    print(f"{'='*50}\n")

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_structure': str(model)
    }
