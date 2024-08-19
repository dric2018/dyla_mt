from config import Config

import re

def char_tokenizer(text):
    return list(text)


def parseText(s):
    s = re.sub(r"[!'&\(\),-./:;=?+.\n\[\]]", r"", s.lower().strip())
    s = re.sub(' +', ' ', s) # remove extra spaces

    return s

def filter_samples(df, filter_col:str="dyu_len"):
    return df[df[filter_col] < Config.MAX_LENGTH].copy()


def prepare_data(df, filter_col:str="dyu_len"):
    df['dyu']       = df['translation.dyu'].apply(lambda x: parseText(x))
    df['fr']        = df['translation.fr'].apply(lambda x: parseText(x))

    # Drop the original column if it's no longer needed
    df              = df.drop(columns=['translation.dyu', 'translation.fr', 'ID'])

# compute sequence lengths

    df['dyu_len']   = df['dyu'].apply(lambda x: len(x))
    df['fr_len']    = df['fr'].apply(lambda x: len(x))

    df              = filter_samples(df, filter_col=filter_col)

    return df.copy()


