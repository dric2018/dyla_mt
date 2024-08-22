from collections import Counter
from config import Config

import logging
logging.basicConfig(level="INFO")

from pprint import pprint

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