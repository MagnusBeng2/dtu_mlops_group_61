from typing import Dict, List, Optional

import pytorch_lightning as pl
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import time

from src.models import _MODEL_PATH


class Model(pl.LightningModule):
    def __init__(
        self, lr: float = 1e-3, batch_size: int = 1, tokenizer_name: str = "t5-small"
    ) -> None:
        """
        A PyTorch Lightning wrapper for the T5-small model.

        Parameters
        ----------
        lr : float
            Learning rate for training. Must be positive.

        batch_size : int
            Batch size for training. Must be greater than 0.

        tokenizer_name : str
            Name of the tokenizer to use (e.g., "t5-small").

        Raises
        ------
        ValueError
            If learning rate or batch size are invalid.
        """
        super().__init__()

        if lr <= 0:
            raise ValueError("Learning rate must be greater than zero!")
        if batch_size <= 0:
            raise ValueError("Batch size must be greater than zero!")

        self.lr = lr
        self.batch_size = batch_size

        # Load tokenizer and model
        self.tokenizer = T5Tokenizer.from_pretrained(
            tokenizer_name, cache_dir=_MODEL_PATH, model_max_length=512, legacy=False
        )
        self.t5 = T5ForConditionalGeneration.from_pretrained(
            tokenizer_name, cache_dir=_MODEL_PATH
        )
        self.save_hyperparameters()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[str]:
        """
        Perform a forward pass for inference.

        Parameters
        ----------
        input_ids : torch.Tensor
            The tokenized input IDs.

        attention_mask : torch.Tensor
            The attention mask.

        Returns
        -------
        List[str]
            List of generated output strings.
        """
        outputs = self.t5.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=20)
        return [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]

    def _shared_step(self, batch: Dict[str, torch.Tensor], step_name: str) -> torch.Tensor:
        """
        Perform a shared step for training, validation, and testing.
        """
        # Extract tensors directly from batch
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Compute loss
        loss = self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        ).loss
        self.log(f"{step_name}_loss", loss, batch_size=self.batch_size)
        return loss


    def training_step(self, batch, batch_idx):
        print(f"Batch {batch_idx} input_ids: {batch['input_ids'].shape}")
        print(f"Batch {batch_idx} labels: {batch['labels'].shape}")
        loss = self._shared_step(batch, "train")
        print(f"Batch {batch_idx} loss: {loss.item()}")
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: Optional[int] = None) -> torch.Tensor:
        loss = self._shared_step(batch, "val")

        # Compute BLEU score
        candidate_corpus = self.forward(batch["input_ids"], batch["attention_mask"])
        references_corpus = [[ref.split()] for ref in batch["labels"]]
        bleu = corpus_bleu(
            references_corpus,
            [cand.split() for cand in candidate_corpus],
            smoothing_function=SmoothingFunction().method1,
        )
        self.log("val_bleu_score", bleu, batch_size=self.batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        # Forward pass for generating predictions
        outputs = self.t5.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=128,
            num_beams=4,  # Adjust as needed
        )

        # Decode the predictions and labels
        decoded_predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
        
        print(f"Predictions: {decoded_predictions}")
        print(f"Labels: {decoded_labels}")

        # Prepare the references and predictions corpora
        references_corpus = [[ref.split()] for ref in decoded_labels]  # Split each label into words
        predictions_corpus = [pred.split() for pred in decoded_predictions]  # Split each prediction into words

        # Calculate BLEU score or other metrics
        bleu_score = self.calculate_bleu(predictions_corpus, references_corpus)

        # Log BLEU score or any other metrics
        self.log("val_bleu", bleu_score, prog_bar=True)

        # Calculate and log the validation loss
        loss = self.loss_function(outputs, batch["labels"])  # Replace with your actual loss function
        self.log("val_loss", loss, prog_bar=True)

        return loss


    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
