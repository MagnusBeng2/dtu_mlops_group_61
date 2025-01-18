import argparse
import os
from typing import Optional
from fastapi import FastAPI
from src.models.model import Model

app = FastAPI()

# Initialize the model as a global variable
model = None


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


# Load the model when the module is imported
model = load_model()


@app.get("/translate/{input}")
def translate(input: str = "Hello world"):
    """
    Translate the input string from English to German using the loaded model.
    """
    if model is None:
        raise RuntimeError("Model is not loaded. Please load the model before making a request.")
    prediction = model.forward(
        input_ids=model.tokenizer(input, return_tensors="pt").input_ids,
        attention_mask=model.tokenizer(input, return_tensors="pt").attention_mask,
    )
    return {"en": input, "de translation": prediction[0]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", action="store_true", help="Run the API server")
    parser.add_argument("--input", type=str, help="Translate the given input string")
    args = parser.parse_args()

    # Load the model
    model = load_model()

    # Handle command-line input for translation
    if args.input:
        prediction = model.forward(
            input_ids=model.tokenizer(args.input, return_tensors="pt").input_ids,
            attention_mask=model.tokenizer(args.input, return_tensors="pt").attention_mask,
        )
        print({"en": args.input, "de translation": prediction[0]})
    elif args.api:
        # Run the API server
        import uvicorn
        uvicorn.run("predict_model:app", host="127.0.0.1", port=8000, reload=True)
    else:
        print("Please specify either --input or --api.")