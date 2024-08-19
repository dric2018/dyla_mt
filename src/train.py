import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from model import DyulaTranslator
from dataset import DyulaDataModule

import wandb

if __name__ == "__main__":

    # Load data
    # TBA
    
    # Start a new W&B run
    wandb.init(project="dyula-french-translation")

    # Initialize Weights & Biases logger
    wandb_logger = WandbLogger(project="dyula-french-translation")

    # Define model and data module (assuming they are defined elsewhere)
    model = DyulaTranslator()
    data_module = DyulaDataModule()

    # Initialize Trainer with GPU support
    trainer = pl.Trainer(gpus=1, max_epochs=10, logger=wandb_logger)  # Use 1 GPU

    # Start training
    trainer.fit(model, data_module)

    # Finish the W&B run
    wandb.finish()