from copy import deepcopy

import datasets
import torch
from pytorch_lightning import Trainer
from tqdm import tqdm
from transformers import T5Tokenizer
from torch.utils.data import DataLoader, TensorDataset

from src.models.model import Model


def test_model_is_torch():
    model = Model()
    assert isinstance(
        next(iter(model.t5.parameters())), torch.Tensor
    )  # To ensure that it runs in torch.


tokenizer = T5Tokenizer.from_pretrained("t5-small")

model = Model()

def test_model_output():
    # Define the raw input sentences
    input = ["The house is wonderful", "I am hungry"]

    tokenized_input = tokenizer(input, return_tensors="pt", padding=True, truncation=True)

    output = model(
        input_ids=tokenized_input["input_ids"],
        attention_mask=tokenized_input["attention_mask"]
    )

    assert isinstance(output, list)
    assert output != []
    assert isinstance(output[0], str)


def test_steps():

    inputs = ["The house is wonderful.", "I am hungry."]
    labels = ["Das Haus ist wunderbar.", "Ich habe hunger."]

    tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
    tokenized_labels = tokenizer(labels, return_tensors="pt", padding=True, truncation=True)

    tokenized_batch = {
    "input_ids": tokenized_inputs["input_ids"],
    "attention_mask": tokenized_inputs["attention_mask"],
    "labels": tokenized_labels["input_ids"],
    }

    loss = model.training_step(tokenized_batch, 1)
    assert isinstance(loss.item(), float)  # loss is given as a float
    assert isinstance(loss, torch.Tensor)  # loss is a torch tensor
    assert not torch.any(torch.isnan(loss)).item()  # loss is not nan

    loss = model.validation_step(tokenized_batch, 1)
    assert isinstance(loss.item(), float)
    assert isinstance(loss, torch.Tensor)
    assert not torch.any(torch.isnan(loss)).item()

    loss_dict = model.test_step(tokenized_batch, 1)
    test_loss = loss_dict["test_loss"]  # Extract the tensor for test loss

    assert isinstance(test_loss.item(), float)  # Check if the loss value is a float
    assert isinstance(test_loss, torch.Tensor)  # Ensure the loss is a tensor
    assert not torch.any(torch.isnan(test_loss)).item()  # Check that the loss is not NaN


def test_training_loop():
    """
        Runs a training loop and checks if the weights change.
    """
    inputs = ["The house is wonderful.", "I am hungry."]
    labels = ["Das Haus ist wunderbar.", "Ich habe hunger."]

    tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
    tokenized_labels = tokenizer(labels, return_tensors="pt", padding=True, truncation=True)

    # Create a dataset as a list of dictionaries
    dataset = [
        {
            "input_ids": tokenized_inputs["input_ids"][i],
            "attention_mask": tokenized_inputs["attention_mask"][i],
            "labels": tokenized_labels["input_ids"][i],
        }
        for i in range(len(inputs))
    ]

    # Use a DataLoader that collates data into dictionaries
    def collate_fn(batch):
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch]),
        }

    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    model = Model(lr=1e-3)
    old_params = deepcopy(model.state_dict())

    # Overfit model in Pytorch Lightning
    trainer = Trainer(
        enable_progress_bar=True,
        enable_checkpointing=False,
        max_epochs=40,
        overfit_batches=1,
        log_every_n_steps=1,
        accelerator="cpu",
    )
    trainer.fit(model, dataloader)
    assert trainer.logged_metrics["train_loss"].item() < 0.1

    new_params = deepcopy(model.state_dict())
    for k in tqdm(old_params.keys()):
        assert torch.any(old_params[k] != new_params[k]).item()


def test_predict_model_commandline():
    import os

    out = os.popen(
        "python "
        + os.path.join("src", "models", "predict_model.py")
        + ' --input="The house is wonderful"'
    ).read()
    assert (
        out
        == "{'en': 'The house is wonderful', 'de translation': 'Das Haus ist wunderbar.'}\n"
    )
