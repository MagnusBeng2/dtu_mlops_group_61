from typing import Dict, List, Optional

import pytorch_lightning as pl
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

import warnings
warnings.filterwarnings("ignore", message="Can't initialize NVML")
warnings.filterwarnings("ignore", message="The torchvision.datapoints and torchvision.transforms.v2 namespaces are still Beta")
warnings.filterwarnings("ignore", message="Passing a tuple of `past_key_values` is deprecated")
import torchvision
torchvision.disable_beta_transforms_warning()

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
            tokenizer_name, cache_dir=_MODEL_PATH, model_max_length=128, legacy=False
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
        self.log(f"{step_name}_loss", loss, batch_size=self.batch_size, logger=True)
        return loss

    def calculate_bleu(self, predictions_corpus: List[List[str]], references_corpus: List[List[List[str]]]) -> float:
        return corpus_bleu(
            references_corpus,
            predictions_corpus,
            smoothing_function=SmoothingFunction().method1,
        )

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        # Generate predictions
        outputs = self.t5.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=128,
            num_beams=4
        )

        # Decode predictions and labels
        decoded_predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

        # Compute BLEU score
        references_corpus = [[ref.split()] for ref in decoded_labels]
        predictions_corpus = [pred.split() for pred in decoded_predictions]
        bleu_score = self.calculate_bleu(predictions_corpus, references_corpus)
        bleu_score = torch.tensor(bleu_score, dtype=torch.float32)  # Convert to torch.float32
        self.log("val_bleu", bleu_score, prog_bar=True)

        # Compute validation loss
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        loss = self.t5(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss

        # Log validation loss as 'val_loss'
        self.log("val_loss", loss, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        # Generate predictions
        outputs = self.t5.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=128,
            num_beams=4,  # Adjust as needed
        )

        # Decode predictions and labels
        decoded_predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

        # Compute BLEU score
        references_corpus = [[ref.split()] for ref in decoded_labels]
        predictions_corpus = [pred.split() for pred in decoded_predictions]
        bleu_score = self.calculate_bleu(predictions_corpus, references_corpus)

        # Compute test loss
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        test_loss = self.t5(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss

        # Log test metrics (ensure each key is logged only once)
        self.log("test_loss", test_loss, prog_bar=True, batch_size=self.batch_size)
        self.log("test_bleu", bleu_score, prog_bar=True, batch_size=self.batch_size)

        return {"test_loss": test_loss, "test_bleu": bleu_score}
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
