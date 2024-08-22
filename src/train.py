from config import Config

from torchinfo import summary

import logging
logging.basicConfig(level='INFO')

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme

from model import DyulaTranslator
from dataset import build_data_module

from utils.utils import train_fn, valid_fn, experiment

import wandb

if __name__ == "__main__":

    # Load data
    dm = build_data_module()

    src_vocab_size = dm.src_tokenizer._get_vocab_size()
    tgt_vocab_size = dm.tgt_tokenizer._get_vocab_size()
    
    # Start a new W&B run
    wandb.init(project="dyula-french-translation")

    # Initialize Weights & Biases logger
    wandb_logger = WandbLogger(
        project="dyula-french-translation", 
        save_dir=Config.LOG_DIR,
        prefix="dyula-fr"
    )

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
    progress_bar = RichProgressBar(
        theme=RichProgressBarTheme(
            # description="Training",
            progress_bar="cyan1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
            metrics_text_delimiter="\n",
            metrics_format=".5f",
        ),
        leave=True
    )
    ckpt_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=Config.MODEL_ZOO,
        filename='dyula_mt'
    )
    trainer = pl.Trainer(
        default_root_dir=Config.LOG_DIR,
        accelerator=Config.device.type, 
        max_epochs=Config.EPOCHS, 
        logger=wandb_logger,
        # logger=True
        callbacks=[progress_bar, ckpt_callback]
    )

    # # Start training
    trainer.fit(model, dm)

    # # Finish the W&B run
    wandb.finish()

