from collections import Counter
from config import Config

import logging
logging.basicConfig(level="INFO")

from utils import char_tokenizer

class Tokenizer:
    def __init__(
            self,
            level:str="word"
        ):
        self.level = level

        # vocabularies
        self.src_vocab = None
        self.tgt_vocab = None

    def encode(self, text):
        # text = parseText(text)

        if self.level =="word":
            return text.split()  # For word-level tokenization
        else:
            return list(text)  # For character-level tokenization

    def yield_tokens(self, data_iter):
        for text in data_iter:
            yield self.encode(text)


    def build_vocab(
            self,
            sentences, 
            min_freq:int=2, 
            specials:list=Config.SPECIAL_TOKENS,
            mt:bool=True,
            kind:str="source"
        ):
        # Tokenize sentences and build frequency dictionary
        token_counts = Counter(token for sentence in sentences for token in char_tokenizer(sentence))
        
        if mt:
            # Filter tokens below min frequency and add specials
            vocab = {token: idx + len(specials) for idx, (token, count) in enumerate(token_counts.items()) if count >= min_freq}
            # Prepend special tokens
            vocab = {tok: idx for idx, tok in enumerate(specials)} | vocab
            # Set default index for unknown tokens
            vocab.setdefault("[UNK]", len(specials))  # This makes "[UNK]" index consistent even if not in specials explicitly
            
            if kind=="source":
                self.src_vocab = vocab
            else:
                self.tgt_vocab = vocab

            