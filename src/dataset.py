import numpy as np

from config import Config

import logging
logging.basicConfig(level='INFO')

import os.path as osp

import pandas as pd

from tokenizer import Tokenizer

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule

import sys

class DyulaDFDataset(Dataset):
    def __init__(
        self,
        df:pd.DataFrame,
        src_tokenizer,
        tgt_tokenizer,
        src_max_len:int=Config.MAX_LENGTH,
        trg_max_len:int=Config.MAX_LENGTH
    ):
        super().__init__()

        self.df             = df
        self.src_tokenizer  = src_tokenizer
        self.tgt_tokenizer  = tgt_tokenizer
        self.src_max_len    = src_max_len
        self.trg_max_len    = trg_max_len

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        source_sentence = self.df.iloc[idx].dyu
        src_len         = self.df.iloc[idx].dyu_len

        target_sentence = self.df.iloc[idx].fr
        tgt_len         = self.df.iloc[idx].fr_len

        source_tokens   = [self.src_tokenizer.vocab[token] for token in self.src_tokenizer.encode(source_sentence)]
        target_tokens   = [self.tgt_tokenizer.vocab[token] for token in self.tgt_tokenizer.encode(target_sentence)]

        source_tokens   = [self.src_tokenizer.vocab["[SOS]"]] + source_tokens + [self.src_tokenizer.vocab["[EOS]"]]
        target_tokens   = [self.tgt_tokenizer.vocab["[SOS]"]] + target_tokens + [self.tgt_tokenizer.vocab["[EOS]"]]

        return (
            torch.tensor(source_tokens, dtype=torch.long), 
            torch.tensor(target_tokens, dtype=torch.long),
            src_len,
            tgt_len
        )
            


# Collate function for padding
def collate_fn(batch):
    source_batch, target_batch, src_lens, tgt_lens = zip(*batch)
    source_batch = pad_sequence(source_batch, padding_value=0, batch_first=True)
    target_batch = pad_sequence(target_batch, padding_value=0, batch_first=True)
    return source_batch, target_batch, torch.tensor(src_lens, dtype=torch.long), torch.tensor(tgt_lens, dtype=torch.long)


class TranslationDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        train_df:pd.DataFrame, 
        val_df:pd.DataFrame, 
        src_tokenizer,
        tgt_tokenizer,
        batch_size=Config.BATCH_SIZE
    ):
        super().__init__()
        self.train_df       = train_df
        self.val_df         = val_df
        self.src_tokenizer  = src_tokenizer
        self.tgt_tokenizer  = tgt_tokenizer
        self.batch_size     = batch_size

    def setup(self, stage=None):
        logging.info(f"Building datasets...")
        self.train_dataset = DyulaDFDataset(
            self.train_df, 
            src_tokenizer=self.src_tokenizer,
            tgt_tokenizer=self.tgt_tokenizer
        )
        num_train_samples = len(self.train_dataset)

        self.val_dataset = DyulaDFDataset(
            self.val_df, 
            src_tokenizer=self.src_tokenizer,
            tgt_tokenizer=self.tgt_tokenizer
        )
        num_val_samples = len(self.val_dataset)

        logging.info(f"> # Training samples: {num_train_samples}")
        logging.info(f"> # Validation samples: {num_val_samples}")


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            collate_fn=collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            collate_fn=collate_fn
        )

def build_data_module()->LightningDataModule:

    logging.info("Loading dataset files")
    train = pd.read_csv(osp.join(Config.DATA_DIR, "preprocessed/train.csv"))
    valid = pd.read_csv(osp.join(Config.DATA_DIR, "preprocessed/valid.csv"))
    # test = pd.read_csv(osp.join(DATA_DIR, "preprocessed/test.csv"))

    source_sentences = train.dyu.values.tolist() + valid.dyu.values.tolist()
    target_sentences = train.fr.values.tolist() + valid.fr.values.tolist()
    
    print()
    logging.info("Building vocabularies")
    
    # source tokenizer
    src_tokenizer = Tokenizer(name="dyu")
    src_tokenizer.build_vocab(source_sentences)
    src_vocab_size = src_tokenizer._get_vocab_size()
    print(f"Vocab size: {src_vocab_size}")

    # target tokenizer
    tgt_tokenizer = Tokenizer(name="fr")
    tgt_tokenizer.build_vocab(target_sentences)
    tgt_vocab_size = tgt_tokenizer._get_vocab_size()
    print(f"Vocab size: {tgt_vocab_size}")

    # Assert conditions to ensure vocabularies match for special tokens
    assert src_tokenizer.vocab["[UNK]"] == tgt_tokenizer.vocab["[UNK]"]
    assert src_tokenizer.vocab["[PAD]"] == tgt_tokenizer.vocab["[PAD]"]

    logging.info("Building Data module")
    
    # Create datasets
    dm = TranslationDataModule(
        train_df=train,
        val_df=valid,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        batch_size=Config.BATCH_SIZE
    )

    return dm


if __name__=="__main__":
    
    dm = build_data_module()
    dm.setup()

    # Training data loader sanity check
    print()
    logging.info("> Train")
    for source_batch, target_batch, src_lens, tgt_lens in dm.train_dataloader():
        print("Source batch:", source_batch.shape)
        print("Target batch:", target_batch.shape)
        print("src lens: ", src_lens)
        print("tgt lens: ", tgt_lens)
        break

    # Validation data loader sanity check
    logging.info("> Validation")
    for source_batch, target_batch, src_lens, tgt_lens in dm.val_dataloader():
        print("Source batch:", source_batch.shape)
        print("Target batch:", target_batch.shape)
        print("src lens: ", src_lens)
        print("tgt lens: ", tgt_lens)
        break

    idx                 = np.random.randint(low=0, high=Config.BATCH_SIZE)
    src                 = source_batch[idx]
    decoded_src         = dm.src_tokenizer.decode(src.tolist())
    print(f"> {decoded_src}")

