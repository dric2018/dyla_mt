from config import Config

from dataset import build_data_module

from model import DyulaTranslator

import logging
logging.basicConfig(level='INFO')

import numpy as np

import os.path as osp

from torchinfo import summary

from utils.utils import calc_edit_distance

if __name__=="__main__":

    dm = build_data_module()
    dm.setup()

    src_vocab_size = dm.src_tokenizer._get_vocab_size()
    tgt_vocab_size = dm.tgt_tokenizer._get_vocab_size()

    print()
    logging.info("Building model")
    
    # load model from ckpt
    model = DyulaTranslator.load_from_checkpoint(
        checkpoint_path=osp.join(Config.MODEL_ZOO, "dyula_mt-v2.ckpt"),
        input_dim=src_vocab_size, 
        output_dim=tgt_vocab_size,
        src_tokenizer=dm.src_tokenizer,
        tgt_tokenizer=dm.tgt_tokenizer
    )
    
    model.to(Config.device)
    model.eval()
    
    # print(model)
    summary(model=model)

    source_batch, target_batch, src_lens, tgt_lens = next(iter(dm.val_dataloader()))

    print("> (source); = (target); < (predictions)")

    idx                 = np.random.randint(low=0, high=Config.BATCH_SIZE)
    src                 = source_batch[idx]
    tgt                 = target_batch[idx]
    
    decoded_src         = dm.src_tokenizer.decode(src.tolist())
    print(f"> {decoded_src}")
    decoded_tgt         = dm.tgt_tokenizer.decode(tgt.tolist())
    print(f"= {decoded_tgt}")

    
    logits, attn_ws = model(source_batch.to(Config.device), trg=None, teacher_forcing_ratio=0.)
    # print(f"predictions: {logits.shape}, attn_w: {attn_ws.shape}")
    decoded_preds       = model.decode_predictions(
        logits[idx], 
        tokenizer=dm.tgt_tokenizer
    ) 
    print(f"< {decoded_preds}")

    dist = calc_edit_distance(
        pred=[decoded_preds], 
        trg=[decoded_tgt], 
        tokenizer=dm.tgt_tokenizer
    )
    print(f"Lev. dist: {dist}")

    # print()
    # model.plot_attention(attn_ws[idx], src_len=src_lens[idx], tgt_len=tgt_lens[idx])