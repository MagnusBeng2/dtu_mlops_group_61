import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import Optional
from fastapi import FastAPI
from src.models.model import Model
from fastapi import HTTPException
import typer

app = FastAPI()
cli = typer.Typer()

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


@app.get("/translate/{input}")
def translate(input: str = "Hello world"):
    global model
    if model is None:
        try:
            model = load_model()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load the model: {str(e)}")

    try:
        # Debugging input and tokenized output
        tokenized = model.tokenizer(input, return_tensors="pt")
        print(f"Tokenized Input: {tokenized}")

        # Debugging prediction output
        prediction = model.forward(
            input_ids=tokenized.input_ids,
            attention_mask=tokenized.attention_mask
        )
        print(f"Model Prediction: {prediction}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

    return {"en": input, "de translation": prediction[0]}

@cli.command()
def main(api: bool = False, input: Optional[str] = None):
    """
    Main entry point for running the script as a command or API.
    """
    global model
    model = load_model()

    if input:
        # Translate a single input string
        prediction = model.forward(
            input_ids=model.tokenizer(input, return_tensors="pt").input_ids,
            attention_mask=model.tokenizer(input, return_tensors="pt").attention_mask,
        )
        print({"en": input, "de translation": prediction[0]})
    elif api:
        # Run the API server
        import uvicorn
        print("Starting the API server...")
        uvicorn.run("src.models.predict_model:app", host="127.0.0.1", port=8000, reload=True)
    else:
        print("Please specify either --input or --api.")


if __name__ == "__main__":
    cli()
