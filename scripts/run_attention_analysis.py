"""Run attention analysis on selected variants."""
from models.load_model import load_model

MODEL_NAME = "zhihan1996/DNABERT-2-117M"

if __name__ == "__main__":
    model, tokenizer = load_model(MODEL_NAME)
    print("Model loaded:", MODEL_NAME)
