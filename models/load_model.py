"""Load pretrained genomic foundation model."""
from transformers import AutoModel, AutoTokenizer

def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    return model, tokenizer
