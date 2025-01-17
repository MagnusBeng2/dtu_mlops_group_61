import argparse
import os
from typing import Optional
from fastapi import FastAPI
from src.models.model import Model

app = FastAPI()


def get_latest_checkpoint(base_dir="lightning_logs"):
    """
    Get the latest model checkpoint from the lightning_logs directory.
    """
    versions = sorted(
        (v for v in os.listdir(base_dir) if v.startswith("version_")),
        key=lambda x: int(x.split("_")[-1]),
        reverse=True,
    )
    for version in versions:
        checkpoints_dir = os.path.join(base_dir, version, "checkpoints")
        if os.path.exists(checkpoints_dir) and os.listdir(checkpoints_dir):
            return os.path.join(checkpoints_dir, os.listdir(checkpoints_dir)[0])
    raise FileNotFoundError(f"No checkpoints found in {base_dir}.")


def load_model():
    """
    Load the model from the latest checkpoint or initialize a fresh model.
    """
    try:
        checkpoint_path = get_latest_checkpoint()
        print(f"Loading model from checkpoint: {checkpoint_path}")
        return Model.load_from_checkpoint(checkpoint_path=checkpoint_path)
    except FileNotFoundError as e:
        print(e)
        print("No checkpoints found. Using an untrained model.")
        return Model()


@app.get("/translate/{input}")
def translate(input: str = "Hello world"):
    """
    Translate the input string from English to German using the loaded model.
    """
    prediction = model.forward(
        input_ids=model.tokenizer(input, return_tensors="pt").input_ids,
        attention_mask=model.tokenizer(input, return_tensors="pt").attention_mask,
    )
    return {"en": input, "de translation": prediction[0]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="English string to be translated")
    args = parser.parse_args()

    # Load the model
    model = load_model()

    # Translate the input
    if args.input:
        result = translate(input=args.input)
        print(result)
    else:
        print("Please provide an input string using the --input argument.")
