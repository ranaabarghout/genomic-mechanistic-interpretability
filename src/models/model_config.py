"""
Model Configuration
==================
Supported models for genomic interpretability analysis.

Use this to switch between models with different capabilities.
"""

MODELS = {
    'dnabert2': {
        'name': 'zhihan1996/DNABERT-2-117M',
        'description': 'DNABERT-2 - Latest genomic model (optimized, no attention outputs)',
        'supports_attention': False,
        'genomic_pretrained': True,
        'tokenizer_type': 'kmer',  # 6-mer
        'max_length': 512,
        'recommended_for': 'hidden_state_analysis'
    },
    'dnabert': {
        'name': 'zhihan1996/DNA_bert_6',
        'description': 'DNABERT (original) - Genomic BERT with full attention support',
        'supports_attention': True,
        'genomic_pretrained': True,
        'tokenizer_type': 'kmer',  # 6-mer
        'max_length': 512,
        'recommended_for': 'mechanistic_attention'
    },
    'bert-base': {
        'name': 'google-bert/bert-base-uncased',
        'description': 'BERT Base - Standard transformer (not genomic)',
        'supports_attention': True,
        'genomic_pretrained': False,
        'tokenizer_type': 'wordpiece',
        'max_length': 512,
        'recommended_for': 'attention_analysis_non_genomic'
    }
}

# Default models by analysis type
DEFAULT_MODEL = 'dnabert2'  # For hidden state analysis
ATTENTION_MODEL = 'dnabert'  # For attention-based analysis


def get_model_config(model_key: str = None):
    """
    Get model configuration.

    Args:
        model_key: One of 'dnabert2', 'bert-base', 'nucleotide-transformer'
                  If None, returns default (dnabert2)

    Returns:
        Dictionary with model configuration
    """
    if model_key is None:
        model_key = DEFAULT_MODEL

    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")

    return MODELS[model_key]


def list_models():
    """Print available models"""
    print("Available models:")
    for key, config in MODELS.items():
        print(f"\n  {key}:")
        print(f"    Name: {config['name']}")
        print(f"    Description: {config['description']}")
        print(f"    Supports attention: {config['supports_attention']}")
        print(f"    Genomic pretrained: {config['genomic_pretrained']}")
