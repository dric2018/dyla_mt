from config import Config

from dataset import TranslationDataModule

from torchinfo import summary
from modules import DyulaTranslator

import logging
logging.basicConfig(level='INFO')

import os.path as osp
import pandas as pd

import lightning as pl
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

from rich.traceback import install 
install()

from utils.utils import generate_experiment_name

import wandb

if __name__ == "__main__":

    # Load data
    # dm = build_data_module()

    exp_name = generate_experiment_name()

    logging.info("Loading dataset files")
    train = pd.read_csv(osp.join(Config.DATA_DIR, "preprocessed/train.csv"))
    valid = pd.read_csv(osp.join(Config.DATA_DIR, "preprocessed/valid.csv"))

    logging.info("Building Data module")
    # Create datasets
    dm = TranslationDataModule(
        train_df=train,
        val_df=valid,
        pretrained_tokenizer=False,
        batch_size=Config.BATCH_SIZE
    )
    dm.setup()
    
    # Start a new W&B run
    # wandb.init(project="dyula-french-translation")

    # # Initialize Weights & Biases logger
    # wandb_logger = WandbLogger(
    #     project="dyula-french-translation", 
    #     save_dir=Config.LOG_DIR,
    #     prefix="dyula-fr"
    # )

    csv_logger = CSVLogger(Config.LOG_DIR, name=exp_name)

    # Define model 
    print()
    logging.info("Building model")

    model = DyulaTranslator()#.to(Config.device)
    
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
        accelerator=Config.DEVICE.type, 
        max_epochs=Config.EPOCHS, 
        logger=csv_logger,
        callbacks=[progress_bar, ckpt_callback]
    )

    # # Start training
    trainer.fit(model=model, datamodule=dm)

    # # Finish the W&B run
    # wandb.finish()

