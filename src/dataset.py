import numpy as np

from config import Config

import logging
logging.basicConfig(level='INFO')

import os.path as osp

import pandas as pd

import tokenizer

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import T5Tokenizer, ByT5Tokenizer

import lightning as pl
from lightning import LightningDataModule

import sys

class DyulaDFDataset(Dataset):
    def __init__(
        self,
        df:pd.DataFrame,
        tokenizer,
        max_len:int=Config.MAX_LENGTH,
        task:str="train"
    ):
        super().__init__()

        self.df             = df
        self.tokenizer      = tokenizer
        self.max_len        = max_len
        self.task           = task

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        # src
        src_txt         = self.df.iloc[idx].dyu
        src_len         = self.df.iloc[idx].dyu_len
        src             = self.tokenizer.encode_plus(src_txt, 
                                return_tensors="pt", 
                                padding="max_length", 
                                truncation=True, 
                                max_length=self.max_len)
        sample = {
            "input_ids": src["input_ids"].squeeze(),
            "attention_mask": src["attention_mask"].squeeze(),
            "src_len": src_len
        }

        if self.task =="train":

            tgt_txt     = self.df.iloc[idx].fr
            tgt_len     = self.df.iloc[idx].fr_len

            dec_in      = self.tokenizer.encode_plus(tgt_txt, 
                            return_tensors="pt", 
                            padding="max_length", 
                            truncation=True, 
                            max_length=self.max_len)
            
            tgt      = torch.cat((dec_in["input_ids"][1:], torch.tensor([Config.PAD_TOKEN_ID])))
            
            sample.update({
                "dec_in": dec_in["input_ids"].squeeze(),
                "labels": tgt.squeeze(),
                "tgt_len":tgt_len
            })
        return sample
            
class HFDyulaDataset(Dataset):
    def __init__(
        self, 
        df, 
        tokenizer, 
        max_length=Config.MAX_LENGTH, 
        is_train:bool=True
    ):
        self.data = df.copy()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dyula_text = self.data.iloc[idx]['dyu']
        french_text = self.data.iloc[idx]['fr']

        # Tokenize input and target texts
        input_encodings = self.tokenizer(dyula_text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        target_encodings = self.tokenizer(french_text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")

        input_ids = input_encodings['input_ids'].squeeze()
        attention_mask = input_encodings['attention_mask'].squeeze()

        sample = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

        if self.is_train:
            labels = target_encodings['input_ids'].squeeze()
            # Replace padding token id's of the labels by -100 to ignore in loss calculation
            labels[labels == self.tokenizer.pad_token_id] = -100
            sample.update({'labels': labels})

        return sample

def get_dataloader(
    df, 
    tokenizer, 
    batch_size=Config.BATCH_SIZE, 
    max_length=Config.MAX_LENGTH,
    is_train:bool=True,
    hf:bool=False
):
    task = "train" if is_train else "test"

    if hf:
        dataset = HFDyulaDataset(df, tokenizer, max_length, is_train)
    else:
        dataset = DyulaDFDataset(df=df, tokenizer=tokenizer, max_length=max_length, task=task)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=Config.NUM_WORKERS)


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
        tokenizer_name:str=Config.BACKBONE_MODEL_NAME,
        pretrained_tokenizer:bool=True,
        batch_size=Config.BATCH_SIZE
    ):
        super().__init__()
        self.train_df       = train_df
        self.val_df         = val_df
        self.pretrained_tokenizer  = pretrained_tokenizer
        if self.pretrained_tokenizer:
            self.tokenizer      = ByT5Tokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer      = tokenizer.ByT5Tokenizer()

        self.batch_size     = batch_size

    def setup(self, stage=None):
        logging.info(f"Building datasets...")
        self.train_dataset = DyulaDFDataset(
            self.train_df, 
            tokenizer=self.tokenizer
        )
        num_train_samples = len(self.train_dataset)

        self.val_dataset = DyulaDFDataset(
            self.val_df, 
            tokenizer=self.tokenizer,
        )
        num_val_samples = len(self.val_dataset)

        logging.info(f"> # Training samples: {num_train_samples}")
        logging.info(f"> # Validation samples: {num_val_samples}")


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            # collate_fn=collate_fn,
            num_workers=Config.NUM_WORKERS
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            # collate_fn=collate_fn,
            num_workers=Config.NUM_WORKERS
        )

def build_data_module()->LightningDataModule:

    logging.info("Loading dataset files")
    train = pd.read_csv(osp.join(Config.DATA_DIR, "preprocessed/train.csv"))
    valid = pd.read_csv(osp.join(Config.DATA_DIR, "preprocessed/valid.csv"))
    # test = pd.read_csv(osp.join(DATA_DIR, "preprocessed/test.csv"))

    # source_sentences = train.dyu.values.tolist() + valid.dyu.values.tolist()
    # target_sentences = train.fr.values.tolist() + valid.fr.values.tolist()
    
    # print()
    # logging.info("Building vocabularies")
    
    # # source tokenizer
    # src_tokenizer = Tokenizer(name="dyu")
    # src_tokenizer.build_vocab(source_sentences)
    # src_vocab_size = src_tokenizer._get_vocab_size()
    # print(f"Vocab size: {src_vocab_size}")

    # # target tokenizer
    # tgt_tokenizer = Tokenizer(name="fr")
    # tgt_tokenizer.build_vocab(target_sentences)
    # tgt_vocab_size = tgt_tokenizer._get_vocab_size()
    # print(f"Vocab size: {tgt_vocab_size}")

    # # Assert conditions to ensure vocabularies match for special tokens
    # assert src_tokenizer.vocab["[UNK]"] == tgt_tokenizer.vocab["[UNK]"]
    # assert src_tokenizer.vocab["[PAD]"] == tgt_tokenizer.vocab["[PAD]"]

    logging.info("Building Data module")
    
    # Create datasets
    dm = TranslationDataModule(
        train_df=train,
        val_df=valid,
        pretrained_tokenizer=False,
        batch_size=Config.BATCH_SIZE
    )

    return dm


if __name__=="__main__":
    
    dm = build_data_module()
    dm.setup()

    # Training data loader sanity check
    print()
    logging.info("> Train")
    for d in dm.train_dataloader():
        print("Source batch:", d['input_ids'].shape)
        print("Target batch:", d["labels"].shape)
        print("Dec in batch:", d["dec_in"].shape)
        print("src lens: ", d["src_len"])
        print("tgt lens: ", d["tgt_len"])
        break

    # Validation data loader sanity check
    logging.info("> Validation")
    for d in dm.val_dataloader():
        print("Source batch:", d['input_ids'].shape)
        print("Target batch:", d["labels"].shape)
        print("src lens: ", d["src_len"])
        print("tgt lens: ", d["tgt_len"])
        break

    idx                 = np.random.randint(low=0, high=Config.BATCH_SIZE)
    src                 = d['input_ids'][idx]
    decoded_src         = dm.tokenizer.decode(src.tolist())
    print(f"> {decoded_src}")

