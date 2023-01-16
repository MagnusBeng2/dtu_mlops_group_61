import pytorch_lightning as pl
import wandb
from datasets import Dataset
from torch.utils.data import DataLoader

from src.models import _DATA_PATH
from src.models.model import Model

if __name__ == "__main__":

    wandb.init(
        project="mlops_exam_project",
        entity="chrillebon",
        config="src/models/config/default_params.yaml",
    )
    
    lr = wandb.config.lr
    epochs = wandb.config.epochs
    batch_size = wandb.config.batch_size

    model = Model(lr=lr, batch_size=batch_size)
    wandb.watch(model, log_freq=100)

    trainset = Dataset.load_from_disk("data/processed/train")
    testset = Dataset.load_from_disk("data/processed/validation")
    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=8)
    testloader = DataLoader(testset, batch_size=batch_size, num_workers=8)

    trainer = pl.Trainer(
        max_epochs=epochs, default_root_dir="")#, accelerator='gpu', devices = [6], strategy="ddp") 

    trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=testloader)

    print("Done!")
