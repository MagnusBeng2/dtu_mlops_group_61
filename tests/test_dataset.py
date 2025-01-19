import os

import pytest
import torch
from datasets import Dataset

from tests import _PATH_DATA


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_dataset_format():
    trainset = Dataset.load_from_disk("data/processed/train")
    testset = Dataset.load_from_disk("data/processed/validation")

    X_train = trainset[0]
    X_test = testset[0]

    assert list(X_train.keys()) == ["input_ids", "attention_mask", "labels"]
    assert isinstance(X_train["input_ids"], torch.Tensor), "input_ids is not a tensor"
    assert isinstance(X_train["attention_mask"], torch.Tensor), "input_ids is not a tensor"
    assert isinstance(X_train["labels"], torch.Tensor), "input_ids is not a tensor"

    assert list(X_test.keys()) == ["input_ids", "attention_mask", "labels"]
    assert isinstance(X_test["input_ids"], torch.Tensor), "input_ids is not a tensor"
    assert isinstance(X_test["attention_mask"], torch.Tensor), "input_ids is not a tensor"
    assert isinstance(X_test["labels"], torch.Tensor), "input_ids is not a tensor"
