from config import Config

from torchinfo import summary

import logging
logging.basicConfig(level='INFO')

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from model import DyulaTranslator
from dataset import build_data_module

from utils import train_fn, valid_fn, experiment

import wandb

if __name__ == "__main__":

    # Load data
    dm = build_data_module()
    # dm.setup()

    src_vocab_size = dm.src_tokenizer._get_vocab_size()
    tgt_vocab_size = dm.tgt_tokenizer._get_vocab_size()
    
    # Start a new W&B run
    # wandb.init(project="dyula-french-translation")

    # Initialize Weights & Biases logger
    # wandb_logger = WandbLogger(project="dyula-french-translation")

    # Define model 
    print()
    logging.info("Building model")

    model = DyulaTranslator(
        input_dim=src_vocab_size, 
        output_dim=tgt_vocab_size,
        src_tokenizer=dm.src_tokenizer,
        tgt_tokenizer=dm.tgt_tokenizer
    )#.to(Config.device)
    
    # print(model)
    summary(model=model)

    # Initialize Trainer with GPU support
    trainer = pl.Trainer(
        accelerator=Config.device.type, 
        max_epochs=Config.EPOCHS, 
        # logger=wandb_logger
    )

    # # Start training
    trainer.fit(model, dm)

    # # Finish the W&B run
    # wandb.finish()

    # optimizer, lr_scheduler = model.configure_optimizers()

    # print(f"Device: {Config.device}")

    # train_loss = train_fn(
    #     model=model, 
    #     data_loader=dm.train_dataloader(),
    #     optimizer=optimizer[0]
    # )