from collections import Counter
from config import Config

import logging
logging.basicConfig(level="INFO")

import numpy as np

from pprint import pprint

import torch

from typing import List, Tuple

from utils.utils import char_tokenizer


class Tokenizer:

    def __init__(
            self,
            name:str="fr",
            max_len:int=Config.MAX_LENGTH,
            level:str="char"
        ):

        self.name           = name
        self.level          = level
        self.eos_token      = Config.eos_token
        self.unk_token      = Config.unk_token 
        self.pad_token      = Config.pad_token
        self.max_len        = max_len

        # vocabularies
        self.vocab          = None
        self.reverse_vocab  = None

    def encode(self, text):
        # text = parseText(text)

        if self.level =="word":
            token_ids = text.split()  # For word-level tokenization
        else:
            token_ids = list(text)  # For character-level tokenization
        
        # Truncating to max_len
        if len(token_ids) > self.max_len:
            token_ids = token_ids[:self.max_len]
        
        return token_ids

    def yield_tokens(self, data_iter):
        for text in data_iter:
            yield self.encode(text)
    
    def decode(self, token_ids: List[int]) -> str:
        decoded = []
        for idx in token_ids[1:-1]:
            # skip eos and pad tokens 
            if idx not in [Config.SPECIAL_TOKENS.index(self.eos_token), Config.SPECIAL_TOKENS.index(self.pad_token)]:
                token = self.reverse_vocab.get(idx, self.unk_token)
                decoded.append(token)
            else:
                break

        return ''.join(decoded)


    def build_vocab(
            self,
            sentences, 
            specials:list=Config.SPECIAL_TOKENS,
            mt:bool=True
        ):
        logging.info(f"Building {self.name} tokenizer...")
        # Tokenize sentences and build frequency dictionary
        token_counts = Counter(token for sentence in sentences for token in char_tokenizer(sentence))
        
        if mt:
            # Filter tokens below min frequency and add specials
            vocab = {token: idx + len(specials) for idx, (token, count) in enumerate(token_counts.items())}
            # Prepend special tokens
            vocab = {tok: idx for idx, tok in enumerate(specials)} | vocab
            # Set default index for unknown tokens
            vocab.setdefault("[UNK]", len(specials))  # This makes "[UNK]" index consistent even if not in specials explicitly
            
            self.vocab          = vocab
            self.reverse_vocab  = {idx: word for word, idx in self.vocab.items()}

    def _get_vocab_size(self):
        assert self.vocab is not None, "Vocabulary is empty. Please, build the vocabulary first."
        return len(self.vocab)
    
    def _print_vocab(self):
        assert self.vocab is not None, "Vocabulary is empty. Please, build the vocabulary first."
        pprint(self.vocab)    

class ByT5Tokenizer:
    def __init__(self):
        self.special_tokens = Config.SPECIAL_TOKENS
        self.vocab_size = Config.VOCAB_SIZE

    def encode(self, text):
        byte_ids = [Config.SOS_TOKEN_ID] + list(text.encode("utf-8")) + [Config.EOS_TOKEN_ID]
        return byte_ids

    def decode(self, byte_ids):
        decoded_parts = []
        for b in byte_ids:
            if b in self.special_tokens:
                # If it's a special token ID, append its string representation
                decoded_parts.append(self.special_tokens[b])
            else:
                # Otherwise, decode it as a regular byte
                decoded_parts.append(chr(b))
        # Join all parts into a single string
        return ''.join(decoded_parts)

    def pad(self, byte_ids, max_length):
        if len(byte_ids) > max_length:
            return byte_ids[:max_length]
        else:
            return byte_ids + [Config.PAD_TOKEN_ID] * (max_length - len(byte_ids))

    def batch_encode(self, texts, max_length):
        return [self.pad(self.encode(text), max_length) for text in texts]

    def batch_decode(self, batch_byte_ids):
        return [self.decode(byte_ids) for byte_ids in batch_byte_ids]

    def encode_plus(
        self, 
        text, 
        max_length:int=512, 
        add_special_tokens=True, 
        padding="max_length", 
        return_attention_mask:bool=True, 
        return_tensors:str=None,
        truncation:bool=False
    ):
        """
        Encode a single text and provide additional information, such as attention mask, truncation, and padding.
        """
        byte_ids = self.encode(text)

        if add_special_tokens:
            byte_ids.append(Config.EOS_TOKEN_ID)

        if truncation and len(byte_ids) > max_length:
            byte_ids = byte_ids[:max_length]

        if padding == "max_length":
            byte_ids = self.pad(byte_ids, max_length)

        attention_mask = [1 if token != Config.PAD_TOKEN_ID else 0 for token in byte_ids]

        if return_tensors == "pt":
            byte_ids = torch.tensor(byte_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        elif return_tensors == "np":
            byte_ids = np.array(byte_ids, dtype=np.int32)
            attention_mask = np.array(attention_mask, dtype=np.int32)

        return {
            "input_ids": byte_ids,
            "attention_mask": attention_mask
        }