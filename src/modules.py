import dataset 
import numpy as np

#torch
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary

from tqdm import tqdm

from transformers import AutoModelForSeq2SeqLM, T5ForConditionalGeneration, ByT5Tokenizer
import tokenizer

# lightning
import lightning as pl

from config import Config
import dataset

import logging
logging.basicConfig(level='INFO')

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

import os.path as osp

from rich.traceback import install 
install()

import random

import sacrebleu


from utils import utils
import wandb

class MultiHeadAttention(nn.Module):
    def __init__(
        self, 
        embed_dim:int=Config.EMBEDDING_DIM, 
        num_heads:int=Config.N_HEADS,
        dropout=Config.ATTN_DROP_RATE
    ):
        super(MultiHeadAttention, self).__init__()

        assert embed_dim % num_heads == 0, f"Embedding dimension must be divisible by the number of heads. Found embed_dim={embed_dim} and num_heads={num_heads}"
        self.embed_dim          = embed_dim
        self.num_heads          = num_heads
        self.head_dim           = embed_dim // num_heads

        self.dropout            = dropout

        self.W_q                = nn.Linear(embed_dim, embed_dim)
        self.W_k                = nn.Linear(embed_dim, embed_dim)
        self.W_v                = nn.Linear(embed_dim, embed_dim)
        self.out_proj           = nn.Linear(embed_dim, embed_dim)

    def split_heads(self, x):
        # x shape: [batch_size, seq_length, embed_dim]
        batch_size, seq_length, embed_dim = x.size()

        # Reshape to [batch_size, seq_length, num_heads, head_dim]
        x = x.view(batch_size, seq_length, self.num_heads, self.head_dim)

        # Transpose to [batch_size, num_heads, seq_length, head_dim]
        x = x.transpose(1, 2)

        return x


    def forward(self, q, enc_out=None, mask=None):
        B, N, C = q.shape        
        # print(f"In: q:{q.shape}")
        
        if enc_out is not None:
            k, v = enc_out, enc_out
        else:
            k, v = q, q      

        q, k, v = self.W_q(q), self.W_k(k), self.W_v(v)
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v) # [batch_size, num_heads, seq_length, head_dim]
        
        if k.ndim < 4:
            k = self.split_heads(k)
            v = self.split_heads(v)

        k = k.permute(0, 1, 3, 2)  # Shape: [batch_size, num_heads, head_dim, seq_len]

        # print(f"Proj: q:{q.shape}, k:{k.shape}, v:{v.shape}")

        # Flash attention: Calculate attention using scaled dot-product in a memory-efficient way
        attn_logits = torch.matmul(q, k) / (self.head_dim ** 0.5)  # Shape: [batch_size, num_heads, seq_len, seq_len]
        # print(f"attn logits: {attn_logits.shape}")

        if mask is not None:
            if mask.ndim < 4:
                expanded_mask = mask[:, None, None, :]
            else:
                expanded_mask = mask
                
            # print(f"exp mask: {expanded_mask.shape}")
            attn_logits = attn_logits.masked_fill(expanded_mask == 0, float('-inf'))
        
        attn_scores = F.softmax(attn_logits, dim=-1)
        attn_scores= F.dropout(attn_scores, self.dropout, self.training)

        # print(f"attn_probs: {attn_probs.shape}, v: {v.shape}")

        attn_w = (attn_scores @ v).transpose(1, 2).reshape(B, -1, C)
        output = self.out_proj(attn_w) # context vector

        return output, attn_w
    
class FeedForward(nn.Module):
    def __init__(
        self, 
        embed_dim:int, 
        hidden_dim:int=Config.D_FF, 
        dropout:float=Config.FF_DROP_RATE
    ):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

class TransformerLayer(nn.Module):
    def __init__(
        self, 
        embed_dim, 
        num_heads, 
        hidden_dim, 
        use_cross_attn=False,
        dropout=Config.ATTN_DROP_RATE
    ):
        super(TransformerLayer, self).__init__()
        
        self.use_cross_attn = use_cross_attn
        self.self_attn      = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1          = nn.LayerNorm(embed_dim)
        self.dropout1       = nn.Dropout(dropout)

        self.cross_attn     = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2          = nn.LayerNorm(embed_dim)
        self.dropout2       = nn.Dropout(dropout)

        self.ffn            = FeedForward(embed_dim, hidden_dim, dropout)
        self.norm3          = nn.LayerNorm(embed_dim) 
        self.dropout3       = nn.Dropout(dropout)

    def forward(self, x, enc_output=None, mask=None, enc_mask=None):

        cross_attn_w = None

        # self attention
        res = x
        x, self_attn_w = self.self_attn(q=x, mask=mask)

        # add & norm
        x = self.dropout1(x)
        x = self.norm1(x + res)

        if self.use_cross_attn and enc_output is not None:
            # cross attn
            x, cross_attn_w = self.cross_attn(q=x, enc_out=enc_output, mask=enc_mask)
            # add & norm
            x = self.dropout2(x)
            x = self.norm2(x + res)
        
        # pointwise FF
        res = x
        x = self.ffn(x)

        # add & norm
        x = self.dropout3(x)
        x = self.norm3(x + res)

        return x, self_attn_w, cross_attn_w     


class TransformerEncoder(nn.Module):
    def __init__(
        self, 
        num_layers, 
        embed_dim, 
        num_heads, 
        hidden_dim, 
        dropout=Config.ENCODER_DROPOUT
    ):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):

        self_attn_Ws = []

        for layer in self.layers:
            x, self_attn_w, _ = layer(x, mask=mask)
            self_attn_Ws.append(self_attn_w)

        self_attn_Ws    = torch.stack(self_attn_Ws, dim=1) 

        if self.num_layers == 1:
            self_attn_Ws    = self_attn_Ws.squeeze(1)

        return x, self_attn_Ws

class TransformerDecoder(nn.Module):
    def __init__(
        self, 
        num_layers, 
        embed_dim, 
        num_heads, 
        hidden_dim, 
        dropout=Config.DECODER_DROPOUT
    ):
        super(TransformerDecoder, self).__init__()
        
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads, hidden_dim, use_cross_attn=True, dropout=dropout) for _ in range(num_layers)
        ])

    def forward(self, x, enc_output, mask=None, enc_mask=None):
        self_attn_Ws = []
        cross_attn_Ws = []

        for layer in self.layers:
            x, self_attn_w, cross_attn_w = layer(x, enc_output, mask=mask, enc_mask=enc_mask)
            self_attn_Ws.append(self_attn_w)
            cross_attn_Ws.append(cross_attn_w)

        self_attn_Ws = torch.stack(self_attn_Ws, dim=1) 
        cross_attn_Ws = torch.stack(cross_attn_Ws, dim=1)


        if self.num_layers == 1:
            self_attn_Ws    = self_attn_Ws.squeeze(1)
            cross_attn_Ws   = cross_attn_Ws.squeeze(1)        
        
        return x, self_attn_Ws, cross_attn_Ws


class ByT5Model(nn.Module):
    def __init__(
        self, 
        vocab_size:int=Config.VOCAB_SIZE, 
        embed_dim:int=Config.EMBEDDING_DIM, 
        num_heads:int=Config.N_HEADS, 
        hidden_dim:int=Config.D_FF, 
        num_encoder_layers:int=Config.NUM_ENCODER_LAYERS, 
        num_decoder_layers:int=Config.NUM_DECODER_LAYERS, 
        enc_dropout:float=Config.ENCODER_DROPOUT,
        dec_dropout:float=Config.DECODER_DROPOUT,
        max_length:int=Config.MAX_LENGTH  # Default max length for autoregressive decoding
    ):
        super(ByT5Model, self).__init__()
        self.encoder        = TransformerEncoder(num_encoder_layers, embed_dim, num_heads, hidden_dim, enc_dropout)
        self.decoder        = TransformerDecoder(num_decoder_layers, embed_dim, num_heads, hidden_dim, dec_dropout)
        self.embed_tokens   = nn.Embedding(vocab_size, embed_dim)
        self.lm_head        = nn.Linear(embed_dim, vocab_size)
        self.loss_fct       = nn.CrossEntropyLoss(ignore_index=Config.PAD_TOKEN_ID)
        self.max_length     = max_length
        self.start_token_id = Config.PAD_TOKEN_ID  # Start token for autoregressive decoding, usually <pad> or <sos>

    def forward(
        self, 
        input_ids, 
        attention_mask=None, 
        decoder_input_ids=None, 
        labels=None
    ):
        # Encode
        x = self.embed_tokens(input_ids)
        # print("emb out: ", x.shape)
        enc_output, _ = self.encoder(x, mask=attention_mask)
        # print("enc out: ", enc_output.shape)

        # # If `decoder_input_ids` is None, perform autoregressive decoding
        if decoder_input_ids is None:
            return self.autoregressive_decode(enc_output, attention_mask)

        # Decode with provided `decoder_input_ids` (typically for training or teacher forcing)
        y = self.embed_tokens(decoder_input_ids)
        dec_output, self_attn_ws, cross_attn_ws = self.decoder(y, enc_output, mask=attention_mask)

        # Project decoder outputs to vocabulary size using the language modeling head
        lm_logits = self.lm_head(dec_output)

        # If labels are provided, calculate the loss
        loss = None
        if labels is not None:
            # Shift logits and labels for teacher forcing
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Cross-entropy loss
            loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return (loss, lm_logits) if loss is not None else lm_logits

    def autoregressive_decode(self, enc_output, attention_mask=None):
        # Start with the initial token as `decoder_input_ids`
        batch_size, seq_len, _ = enc_output.shape
        decoder_input_ids = torch.full(
            (batch_size, 1), 
            self.start_token_id, 
            dtype=torch.long, 
            device=enc_output.device
        )  # Shape: [batch_size, 1]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=enc_output.device)

        for t in range(self.max_length):
            # Embed the current decoder input
            y = self.embed_tokens(decoder_input_ids)

            # Generate causal mask for the current sequence length
            causal_mask     = utils.generate_causal_mask(decoder_input_ids.size(1)).to(y.device)  # Shape: [seq_len, seq_len]
            batched_mask    = causal_mask.unsqueeze(0).expand(batch_size, -1, -1, -1)
            # print(f"causal_mask: {causal_mask.shape}, batched_mask: {batched_mask.shape}, dec inp:{y.shape}")

            # Decode with encoder outputs and causal mask
            dec_output, self_attn_ws, cross_attn_ws = self.decoder(y, enc_output[:, :y.size(1), :], mask=batched_mask)

            # Project to vocabulary size and get the logits for the last token
            lm_logits = self.lm_head(dec_output)
            next_token_id = lm_logits[:, -1, :].argmax(dim=-1, keepdim=True)  # Get the most probable token
            
            # Check which sequences are finished
            just_finished = (next_token_id == Config.EOS_TOKEN_ID).squeeze(-1)
            finished |= just_finished
            
            # Prevent finished sequences from changing tokens after finishing
            next_token_id[finished] = Config.PAD_TOKEN_ID  # Use pad_token_id for already finished sequences
        

            # Append the new token to `decoder_input_ids`
            decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=1)
            # print(f"cat decoder_input_ids: {decoder_input_ids.shape}")

            # Stop decoding if all sequences have generated the EOS token
            if torch.all(finished):
                break

        return decoder_input_ids[:, 1:]  # Return the generated sequence

class DyulaTranslator(pl.LightningModule):
    def __init__(
        self, 
        model_name=Config.BACKBONE_MODEL_NAME,
        is_pretrained:bool=Config.IS_PRETRAINED
    ):
        super(DyulaTranslator, self).__init__()

        self.is_pretrained = is_pretrained

        if is_pretrained:
            self.translator = T5ForConditionalGeneration.from_pretrained(model_name)
            self.tokenizer = ByT5Tokenizer.from_pretrained(model_name)
        else:
            self.translator = ByT5Model()

        self.learning_rate = Config.LR

    def forward(self, input_ids, attention_mask, decoder_input_ids=None, labels=None):
        if self.is_pretrained:
            output = self.translator(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                decoder_input_ids=decoder_input_ids,
                labels=labels
            )
            return output.loss, output.logits
        else:
            out = self.translator(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            decoder_input_ids=decoder_input_ids, 
            labels=labels
            )
            return out

    def training_step(self, batch, batch_idx):

        if self.is_pretrained:
            decoder_input_ids = batch['labels']
            decoder_input_ids = torch.cat([torch.full((decoder_input_ids.size(0), 1), self.tokenizer.pad_token_id, dtype=torch.long), decoder_input_ids[:, :-1]], dim=1)
            loss, _ = self(
                batch['input_ids'], 
                batch['attention_mask'],
                decoder_input_ids, 
                batch['labels']
            )
        else:
            loss, out = self(
            input_ids=batch["input_ids"], 
            attention_mask=batch["attention_mask"], 
            decoder_input_ids=batch["dec_in"], 
            labels=batch["labels"]
            )

            self.log("train_loss", loss, on_epoch=True, logger=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.is_pretrained:
            decoder_input_ids = batch['labels']
            decoder_input_ids = torch.cat([torch.full((decoder_input_ids.size(0), 1), self.tokenizer.pad_token_id, dtype=torch.long), decoder_input_ids[:, :-1]], dim=1)
            
            generated_ids = self(
                batch['input_ids'], 
                batch['attention_mask'],
                decoder_input_ids,
                batch['labels']
            )
            pred_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            target_text = self.tokenizer.batch_decode(batch['labels'].squeeze(), skip_special_tokens=True)
            
            # compute BLEU score
            bleu_score = sacrebleu.corpus_bleu(pred_text, [target_text])
            self.log('val_bleu', bleu_score.score, on_step=False, on_epoch=True, prog_bar=True)

        # Append random samples to file
        if random.random() < 0.1:  # Append about 10% of validation samples
            with open(osp.join('translations_sample.txt'), 'a') as file:
                for dyu, pred, fr in zip(batch['input_ids'], pred_text, target_text):
                    dyu_sentence = self.tokenizer.decode(dyu, skip_special_tokens=True)
                    file.write(f"dyu: {dyu_sentence}\n")
                    file.write(f"pred: {pred}\n")
                    file.write(f"fr: {fr}\n")
                    file.write("---\n")

            return {"val_loss": val_loss, "BLEU": bleu_score.score}
        else:
    
            val_loss, out = self(
            input_ids=batch["input_ids"], 
            attention_mask=batch["attention_mask"], 
            decoder_input_ids=batch["dec_in"], 
            labels=batch["labels"]
            )
            self.log("val_loss", val_loss, on_epoch=True, logger=True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


if __name__ == '__main__':
    
    dm = dataset.build_data_module()
    dm.setup()

    if Config.IS_PRETRAINED:
        tok = ByT5Tokenizer()
    else:
        tok   = tokenizer.ByT5Tokenizer()

    translator  = DyulaTranslator(is_pretrained=Config.IS_PRETRAINED)
    print(translator)
    summary(translator)

    d           = next(iter(dm.train_dataloader()))
    print("Source batch:", d['input_ids'].shape)
    print("Source batch[0]:", d['input_ids'][0])
    print("Dec in batch:", d["dec_in"].shape)
    print("Attn msk:", d['attention_mask'].shape)
    print("Target batch:", d["labels"].shape)
    print("src lens: ", d["src_len"])
    print("tgt lens: ", d["tgt_len"])
    
    loss, out   = translator(
                    input_ids=d["input_ids"], 
                    attention_mask=d["attention_mask"], 
                    decoder_input_ids=d["dec_in"],
                    labels=d["labels"]
                )

    print(out.shape)
    print(tok.batch_decode(out.argmax(dim=-1)))
    print(f"loss={loss}")

