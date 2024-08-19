from config import Config

import logging
logging.basicConfig(level='INFO')

import os.path as osp

import pandas as pd

from tokenizer import Tokenizer

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class DyulaDFDataset(Dataset):
    def __init__(
        self,
        df,
        tokenizer
    ):
        super().__init__()

        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        source_sentence = self.df.iloc[idx].dyu
        target_sentence = self.df.iloc[idx].fr

        source_tokens = [self.tokenizer.src_vocab[token] for token in self.tokenizer.encode(source_sentence)]
        target_tokens = [self.tokenizer.tgt_vocab[token] for token in self.tokenizer.encode(target_sentence)]

        source_tokens = [self.tokenizer.src_vocab["[SOS]"]] + source_tokens + [self.tokenizer.src_vocab["[EOS]"]]
        target_tokens = [self.tokenizer.tgt_vocab["[SOS]"]] + target_tokens + [self.tokenizer.tgt_vocab["[EOS]"]]

        return torch.tensor(source_tokens, dtype=torch.long), torch.tensor(target_tokens, dtype=torch.long)


# Collate function for padding
def collate_fn(batch):
    source_batch, target_batch = zip(*batch)
    source_batch = pad_sequence(source_batch, padding_value=0)
    target_batch = pad_sequence(target_batch, padding_value=0)
    return source_batch, target_batch


if __name__=="__main__":
    logging.info("Loading dataset files")
    train = pd.read_csv(osp.join(Config.DATA_DIR, "preprocessed/train.csv"))
    valid = pd.read_csv(osp.join(Config.DATA_DIR, "preprocessed/valid.csv"))
    # test = pd.read_csv(osp.join(DATA_DIR, "preprocessed/test.csv"))

    source_sentences = train.dyu.values.tolist() + valid.dyu.values.tolist()
    target_sentences = train.fr.values.tolist() + valid.fr.values.tolist()
    
    logging.info("Building vocabularies")
    
    tokenizer = Tokenizer()
    tokenizer.build_vocab(source_sentences, kind="source")
    tokenizer.build_vocab(target_sentences, kind="target")

    # Assert conditions to ensure vocabularies match for special tokens
    assert tokenizer.src_vocab["[UNK]"] == tokenizer.tgt_vocab["[UNK]"]
    assert tokenizer.src_vocab["[PAD]"] == tokenizer.tgt_vocab["[PAD]"]

    # Get indices for unknown and padding tokens
    unk_index = tokenizer.src_vocab["[UNK]"]
    pad_index = tokenizer.src_vocab["[PAD]"]
    pad_index = tokenizer.src_vocab[Config.pad_token]

    logging.info("Building Data loaders")
    
    # Create datasets
    train_dataset = DyulaDFDataset(
        df=train,
        tokenizer=tokenizer
    )

    valid_dataset = DyulaDFDataset(
        df=valid,
        tokenizer=tokenizer
    )

    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size  = Config.BATCH_SIZE,
        collate_fn  = collate_fn,
        pin_memory  = True,
        shuffle     = True
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size  = Config.BATCH_SIZE,
        collate_fn  = collate_fn
    )

    # Training data loader sanity check
    logging.info("> Training samples")
    for source_batch, target_batch in train_dataloader:
        print("Source batch:", source_batch.shape)
        print("Target batch:", target_batch.shape)
        break

    # Validation data loader sanity check
    logging.info("> Validation samples")
    for source_batch, target_batch in valid_dataloader:
        print("Source batch:", source_batch.shape)
        print("Target batch:", target_batch.shape)
        break
