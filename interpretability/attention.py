"""Attention visualization utilities."""
import torch

def extract_attention(model, inputs):
    outputs = model(**inputs)
    return outputs.attentions
