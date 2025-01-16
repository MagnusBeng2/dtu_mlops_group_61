from typing import Dict, List, Optional

import pytorch_lightning as pl
import torch
from torchtext.data.metrics import bleu_score
from transformers import T5ForConditionalGeneration, T5Tokenizer

from src.models import _MODEL_PATH


class Model(pl.LightningModule):
    def __init__(
        self, lr: Optional[float] = 1e-3, batch_size: Optional[int] = 1, *args, **kwargs
    ) -> None:
        """
        Models are obtained using the code from:
            https://huggingface.co/docs/transformers/model_doc/t5
        Model and tokenizer are loaded from pretrained, per the above link.
        Then model is set to the relevant device ('cuda' if available) and it
        is assigned a learning rate and batch size based on input parameters.

        Parameters
        ----------
        lr : [float integer], optional
            Learning rate for the training of this model. Must be a positive value!

        batch_size : [integer, float], optional
            Batch size for training this model. Must be strictly greater than 0.
            (Any batch size of type float will be cast to an integer!)

        Raises
        ------
        TypeError
            If the learning rate is either not an integer nor a float.
        ValueError
            If the learning rate isn't positive.
        TypeError
            If the batch size is not an integer.
        ValueError
            If the batch size is less than or equal to zero.
        """

        super().__init__(*args, **kwargs)

        if not isinstance(lr, (float, int)):
            raise TypeError("Learning rate must be either an integer or a float.")
        if lr <= 0:
            raise ValueError("Learning rate must be greater than zero!")

        if not isinstance(batch_size, int):
            raise TypeError("Batch size must be an integer.")
        if batch_size <= 0:
            raise ValueError("Batch size must be greater than 0!")

        self.tokenizer = T5Tokenizer.from_pretrained(
            "t5-small", cache_dir=_MODEL_PATH, model_max_length=512
        )

        self.t5 = T5ForConditionalGeneration.from_pretrained(
            "t5-small", cache_dir=_MODEL_PATH
        )
        self.add_module("t5", self.t5)
        self.lr = lr
        self.batch_size = batch_size

    def forward(self, x: List[str]) -> List[str]:
        """
        Perform a forward pass with the model.

        https://huggingface.co/docs/transformers/model_doc/t5#inference
        """
        input_ids = self.tokenizer(
            x, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).input_ids.to(self.t5.device)

        # Forward pass
        outputs = self.t5.generate(input_ids=input_ids, max_new_tokens=20)

        return [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]

    def _inference_training(
        self, batch: Dict[str, Dict[str, List[str]]], batch_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Perform training inference.

        From https://huggingface.co/docs/transformers/model_doc/t5#training
        """
        data = batch["translation"]["en"]
        labels = batch["translation"]["de"]
        encoding = self.tokenizer(
            data,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=512,
        ).to(self.t5.device)
        target_encoding = self.tokenizer(
            labels, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).input_ids.to(self.t5.device)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        loss = self.t5(
            input_ids=input_ids, attention_mask=attention_mask, labels=target_encoding,
        )
        return loss.loss

    def training_step(
        self, batch: List[str], batch_idx: Optional[int] = None
    ) -> torch.Tensor:
        loss = self._inference_training(batch, batch_idx)
        self.log("train_loss", loss, batch_size=self.batch_size)
        return loss

    def validation_step(
        self, batch: Dict[str, Dict[str, List[str]]], batch_idx: Optional[int] = None
    ) -> torch.Tensor:
        loss = self._inference_training(batch, batch_idx)
        self.log("val_loss", loss, batch_size=self.batch_size)

        # BLEU score computation
        candidate_corpus = [self.forward(batch["translation"]["en"])]
        references_corpus = [[ref.split()] for ref in batch["translation"]["de"]]
        bleu = bleu_score(candidate_corpus, references_corpus, max_n=4)
        self.log("val_bleu_score", bleu)

        return loss

    def test_step(
        self, batch: Dict[str, Dict[str, List[str]]], batch_idx: Optional[int] = None
    ) -> torch.Tensor:
        loss = self._inference_training(batch, batch_idx)
        self.log("test_loss", loss, batch_size=self.batch_size)

        # BLEU score computation
        candidate_corpus = [self.forward(batch["translation"]["en"])]
        references_corpus = [[ref.split()] for ref in batch["translation"]["de"]]
        bleu = bleu_score(candidate_corpus, references_corpus, max_n=4)
        self.log("test_bleu_score", bleu)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)
