"""Load pretrained genomic foundation model and fine-tuned checkpoints."""
import torch
from transformers import AutoTokenizer, AutoModel, BertModel, BertTokenizer
from pathlib import Path
import sys
from typing import Optional, Tuple

# Global cache for loaded models to avoid reloading
_MODEL_CACHE = {}


class DNABERTTokenizerWrapper:
    """Wrapper for DNABERT tokenizer that handles k-mer conversion."""

    def __init__(self, tokenizer, kmer_size=6):
        self.tokenizer = tokenizer
        self.kmer_size = kmer_size

    def convert_to_kmers(self, sequence):
        """Convert DNA sequence to space-separated k-mers."""
        kmers = []
        for i in range(len(sequence) - self.kmer_size + 1):
            kmers.append(sequence[i:i+self.kmer_size])
        return ' '.join(kmers)

    def __call__(self, text, **kwargs):
        """Tokenize DNA sequence (convert to k-mers first)."""
        if isinstance(text, str):
            text = self.convert_to_kmers(text)
        elif isinstance(text, list):
            text = [self.convert_to_kmers(s) for s in text]
        return self.tokenizer(text, **kwargs)

    def tokenize(self, text):
        """Tokenize DNA sequence."""
        if isinstance(text, str):
            text = self.convert_to_kmers(text)
        return self.tokenizer.tokenize(text)

    def __getattr__(self, name):
        """Delegate other methods to underlying tokenizer."""
        return getattr(self.tokenizer, name)

def load_base_model(model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu", use_standard_bert: bool = False):
    """Load base genomic or standard BERT model.

    Args:
        model_name: Model name or path
                   - 'zhihan1996/DNABERT-2-117M' for DNABERT-2 (genomic, no attention outputs)
                   - 'google-bert/bert-base-uncased' for standard BERT (full attention support)
        device: Device to load model on
        use_standard_bert: If True, use standard BERT APIs (enables attention outputs)

    Returns:
        Tuple of (model, tokenizer, config)
    """
    print(f"Loading base model: {model_name}")

    # Detect if this is standard BERT
    if 'bert-base' in model_name.lower() or 'bert-large' in model_name.lower():
        use_standard_bert = True

    # Check if already loaded
    cache_key = f"{model_name}_{device}_{use_standard_bert}"
    if cache_key in _MODEL_CACHE:
        print(f"  Using cached model")
        return _MODEL_CACHE[cache_key]

    if use_standard_bert:
        # Load standard BERT (supports attention outputs)
        print("  Using standard BERT API (attention outputs enabled)")
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
    else:
        # Load DNABERT-2 or other custom models
        print("  Using AutoModel API (may not support attention outputs)")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        # Wrap tokenizer for DNABERT models (need k-mer conversion)
        if 'DNA_bert' in model_name or 'DNABERT' in model_name:
            print("  Wrapping tokenizer with k-mer conversion (k=6)")
            tokenizer = DNABERTTokenizerWrapper(tokenizer, kmer_size=6)

    # Enable attention and hidden states output (may not work for all models)
    if hasattr(model.config, 'output_attentions'):
        model.config.output_attentions = True
    if hasattr(model.config, 'output_hidden_states'):
        model.config.output_hidden_states = True

    model.to(device)
    model.eval()

    config = model.config
    print(f"Model loaded on {device}")
    if hasattr(config, 'num_hidden_layers') and hasattr(config, 'num_attention_heads'):
        print(f"Config: {config.num_hidden_layers} layers, {config.num_attention_heads} heads")

    # Cache for reuse
    result = (model, tokenizer, config)
    _MODEL_CACHE[cache_key] = result

    return result


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
