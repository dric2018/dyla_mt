import numpy as np

#torch
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary

from tqdm import tqdm

# lightning
import pytorch_lightning as pl

from config import Config
import dataset

import logging
logging.basicConfig(level='INFO')

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from rich.traceback import install 
install()

from utils import utils
import wandb

class EmbeddingLayer(nn.Module):
    def __init__(
            self, 
            vocab_size:int, 
            emb_dim:int=Config.EMBEDDING_DIM
    ):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(
                            num_embeddings=vocab_size, 
                            embedding_dim=emb_dim, 
                            padding_idx=Config.SPECIAL_TOKENS.index("[PAD]"),
                            device=Config.device
                        )
    
    def forward(self, x):
        return self.embedding(x)
    
class Encoder(nn.Module):
    def __init__(
            self, 
            input_dim:int,
            emb_dim:int=Config.ENCODER_EMBEDDING_DIM, 
            hid_dim:int=Config.ENCODER_HIDDEN_DIM, 
            n_layers:int=Config.NUM_ENCODER_LAYERS, 
            dropout:float=Config.ENCODER_DROPOUT
    ):
        super(Encoder, self).__init__()
        self.src_emb_layer      = EmbeddingLayer(emb_dim=Config.ENCODER_EMBEDDING_DIM, vocab_size=input_dim)
        self.rnn                = nn.LSTM(
                                    emb_dim, 
                                    hid_dim, 
                                    n_layers, 
                                    dropout=dropout, 
                                    batch_first=True,
                                    device=Config.device
                                )
        self.dropout            = nn.Dropout(dropout)
    
    def forward(self, src):
        src_emb                 = self.src_emb_layer(src)
        _, (hidden, cell)       = self.rnn(self.dropout(src_emb))
        return src_emb, hidden, cell

class Decoder(nn.Module):
    def __init__(
            self,
            output_dim:int,
            emb_dim:int=Config.DECODER_EMBEDDING_DIM, 
            hid_dim:int=Config.DECODER_HIDDEN_DIM, 
            n_layers:int=Config.NUM_DECODER_LAYERS, 
            dropout:float=Config.DECODER_DROPOUT
    ):
        super(Decoder, self).__init__()
        
        self.rnn            = nn.LSTM(
                                emb_dim, 
                                hid_dim, 
                                n_layers, 
                                dropout=dropout, 
                                batch_first=True, 
                                device=Config.device
                            )
        self.fc_out         = nn.Linear(hid_dim, output_dim)
        self.dropout        = nn.Dropout(dropout)
    
    def forward(
            self, 
            trg_emb, 
            hidden, 
            cell
    ):
        outputs, (hidden, cell) = self.rnn(self.dropout(trg_emb), (hidden, cell))
        predictions = self.fc_out(outputs)
        
        return predictions, hidden, cell


class ScaledDotProductAttention(nn.Module):
    def __init__(
            self, 
            d_model:int=Config.D_MODEL, 
            dropout:float=0.1
        ):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = d_model ** 0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # print(f"Q: {query.shape}")
        # print(f"V: {value.shape}")
        # print(f"K: {key.shape}")

        # Compute attention scores
        attn_scores = torch.bmm(query, key.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        # attn_weights = self.dropout(attn_weights)
        
        # Compute context vectors
        output = torch.matmul(attn_weights, value)
        return output, attn_weights


class Seq2SeqWithAttention(nn.Module):
    def __init__(
        self, 
        input_dim,
        output_dim,
        max_trg_len:int=Config.MAX_OUTPUT
    ):
        super().__init__()

        self.sos_idx            = Config.SPECIAL_TOKENS.index("[SOS]")
        self.eos_idx            = Config.SPECIAL_TOKENS.index("[EOS]")
        self.max_trg_len        = max_trg_len
        self.output_dim         = output_dim

        self.encoder            = Encoder(input_dim=input_dim)
        self.attention          = ScaledDotProductAttention()

        self.decoder_emb_layer  = EmbeddingLayer(emb_dim=Config.DECODER_EMBEDDING_DIM, vocab_size=output_dim)
        self.decoder            = Decoder(output_dim=Config.DECODER_HIDDEN_DIM)
        
        self.hidden_to_out      = nn.Linear(
                                    in_features=Config.D_MODEL+Config.DECODER_HIDDEN_DIM, 
                                    out_features=Config.DECODER_EMBEDDING_DIM
                                )
        self.activation         = torch.nn.ReLU()
        self.out_layer          = torch.nn.Linear(in_features=Config.DECODER_EMBEDDING_DIM, out_features=output_dim)

        # weight tying
        self.out_layer.weight = self.decoder_emb_layer.embedding.weight  # Weight tying

    def draw_from_dist(self, x):
        # print(f"mpl in: {x.shape}")
        x = self.out_layer(self.activation(self.hidden_to_out(x)))
        # print(f"mpl out: {x.shape}")
        return x
    
    def greedy_decoding(
            self,
            src_embed,
            timesteps,
            trg,
            teacher_forcing_ratio:float=0.
    ):
        
        """
            Greedy decoding for single sample
        """
        # print(f"src emb: {src_embed.shape}")

        dec_inp         = torch.empty(1, 1, dtype=torch.long, device=src_embed.device).fill_(self.sos_idx)

        dec_hidden      = torch.zeros((Config.NUM_DECODER_LAYERS, 1, Config.DECODER_HIDDEN_DIM)).to(Config.device)
        dec_cell        = torch.zeros((Config.NUM_DECODER_LAYERS, 1, Config.DECODER_HIDDEN_DIM)).to(Config.device)
                
        # print(f"dec_hidden: {dec_hidden.shape}")
        # print(f"dec_cell: {dec_cell.shape}")

        for t in range(timesteps):
            # print(f"\n> t: {t}")
            # print(f"dec_inp: {dec_inp.shape}")
            
            dec_emb         = self.decoder_emb_layer(dec_inp)#.unsqueeze(1)
            
            if torch.rand(1).item() < teacher_forcing_ratio:
                if trg is not None:
                    dec_emb = self.decoder_emb_layer(trg[t]) #.unsqueeze(1)
            
            # print(f"dec_emb: {dec_emb.shape}")
            out, dec_hidden, dec_cell = self.decoder(dec_emb, dec_hidden, dec_cell)
            # print(f"dec out: {out.shape}, h: {dec_hidden.shape}")
            
            context, attn_w = self.attention(query=out, key=src_embed.unsqueeze(0), value=src_embed.unsqueeze(0))
            # print(f"ctx: {context.shape}, attn_w: {attn_w.shape}")
            
            concat = torch.cat([out, context], dim=-1)
            # print(f"concat: {concat.shape}")

            probs = self.draw_from_dist(concat[:, -1]).squeeze(1)
            # print(f"probs: {probs.shape}")
            
            _, next_tok = torch.max(probs, dim=-1)
            # print(f"next_tok: {next_tok.shape}, {next_tok}")

            if t >= 1 and next_tok == self.sos_idx:
                break
            
            # update decoder input
            dec_inp = torch.cat((dec_inp, next_tok.unsqueeze(1)), dim=1)

        # return final decoder inp as complete decoded seq along with its aggregated atttention matrix
        return dec_inp.squeeze(0), attn_w.squeeze(0)

    def forward(
            self, 
            src, 
            trg=None, 
            teacher_forcing_ratio:float=1.0
        ):
        # print(f"src: {src.shape}")
        # print(f"trg: {trg.shape}")

        timesteps       = trg.size(1) if trg is not None else self.max_trg_len
        batch_size      = src.size(0)
        output_dim      = self.output_dim        
        # print(f"B: {batch_size}, T: {timesteps}")
        # encode
        # print()
        # print("Encoding inputs...")
        src_emb, enc_hidden, enc_cell = self.encoder(src)
        # print(f"emb: {src_emb.shape}")
        # print(f"hidden: {enc_hidden.shape}")
        # print(f"cell: {enc_cell.shape}")

        #decode
        # print()
        # print("Decoding states...")
        # outputs         = torch.zeros(batch_size, timesteps, output_dim).to(Config.device)
        
        # dec_inp         = torch.full((batch_size,), fill_value=self.sos_idx, dtype=torch.long).to(Config.device)

        # dec_hidden      = enc_hidden
        # dec_cell        = enc_cell
        
        batch_logits    = []
        batch_attn      = []
        # print(f"dec_inp: {dec_inp.shape}")
        # print(f"dec_hidden: {dec_hidden.shape}")
        # print(f"dec_cell: {dec_cell.shape}")

        decoding_lpoop = range(batch_size)
        # if Config.STAGE =="debug":
        #     decoding_lpoop = tqdm(range(batch_size), desc="batch greedy decoding", colour="cyan")

        for b in decoding_lpoop:
            logits, attn_ws = self.greedy_decoding(src_emb[b], timesteps, teacher_forcing_ratio)
            # print(f"logits: {logits.shape}, attn_ws: {attn_ws.shape}")
            batch_logits.append(logits)
            batch_attn.append(attn_ws)

        batch_logits = torch.stack(batch_logits, dim=0)
        print(f"batch_logits: {batch_logits.shape}")

        batch_attn = torch.stack(batch_attn, dim=0)
        print(f"batch_attn: {batch_attn.shape}")
        
        return batch_logits[:, 1:], batch_attn


class DyulaTranslator(pl.LightningModule):
    def __init__(
        self, 
        input_dim,
        output_dim,
        src_tokenizer,
        tgt_tokenizer
    ):
        super().__init__()
        
        # Initialization model
        self.translator             = Seq2SeqWithAttention(
            input_dim=input_dim, 
            output_dim=output_dim
        )

        self.src_tokenizer          = src_tokenizer
        self.tgt_tokenizer          = tgt_tokenizer
        self.criterion              = nn.CrossEntropyLoss(ignore_index=tgt_tokenizer.vocab['[PAD]'])
        self.tf                     = Config.tf_ratio_start  # Start with full teacher forcing
    
    def forward(
            self, 
            src, 
            trg, 
            teacher_forcing_ratio:float=Config.tf_ratio_start
    ):
        
        logits, attn_ws = self.translator(src, trg, teacher_forcing_ratio)
        return logits, attn_ws

    def training_step(
        self, 
        batch, 
        batch_idx
    ):
        src, trg, _, _ = batch
        output, attn_weights = self(src=src, trg=trg, teacher_forcing_ratio=self.tf)
        
        output = output[:, 1:].reshape(-1, output.shape[-1])
        trg = trg[:, 1:].reshape(-1)
        
        loss = self.criterion(output, trg)

        # log train loss
        self.log(
            "train_loss", 
            loss, 
            prog_bar=True, 
            logger=True, 
            on_step=True, 
            on_epoch=True
        )

        # log epoch LR
        final_lr_epoch = float(self.optimizers().param_groups[0]['lr'])
        self.log(
            "lr", 
            final_lr_epoch, 
            prog_bar=True, 
            logger=True, 
            on_step=True, 
            on_epoch=True
        )        
        
        return loss
    
    def validation_step(
            self, 
            batch, 
            batch_idx
    ):
        
        batch_idx_to_display = np.random.randint(low=1, high=Config.BATCH_SIZE)
        src, trg, _, _ = batch
        output, attn_weights = self(src=src, trg=trg, teacher_forcing_ratio=Config.tf_ratio_end)
        n_tgt_tokens = trg.size(-1)
        # print(f"output: {output.shape}, tgt: {trg.shape}")

        output_ = output[:, 1:n_tgt_tokens].reshape(-1, output.shape[-1])
        # print(f"output_: {output_.shape}")
        trg_ = trg[:, 1:].reshape(-1)

        # Calculate loss
        loss = self.criterion(output_, trg_)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        # Log predictions and attention weights to W&B
        if batch_idx % batch_idx_to_display == 0:  # Adjust frequency as needed
            self.log_predictions_and_attention_to_wandb(src, output, trg, attn_weights, batch_idx)

        return loss
    
    # def on_train_epoch_end(self):   
        
    #     all_preds = torch.stack(self.training_step_outputs)
    #     all_labels = torch.stack(self.training_step_targets)
        
    #     rand_idx = np.random.randint(all_preds.shape[0])

    #     # decode predictions
    #     pred = self.decode_predictions(
    #         predicted_ids=all_preds[rand_idx].unsqueeze(0)
    #     )[0]
    #     label = self.decode_predictions(
    #         predicted_ids=all_labels[rand_idx].unsqueeze(0)
    #     )[0]
    #     # log decoded sentenses
    #     with open(config.LOGGING_FILE, "a") as f:            
    #         f.write(f"Epoch #{self.current_epoch}\n")
    #         f.write(f"Train \n")
    #         cer = self.cer_fn(pred, label).item()
    #         wer = self.wer_fn(pred, label).item()
    #         f.write(f"Predicted \t: {pred}\n")
    #         f.write(f"Actual \t\t: {label}\n")
    #         f.write(f"CER \t\t: {cer:.4f}\n")
    #         f.write(f"WER \t\t: {wer:.4f}\n\n")      
            
    #     # free mem
    #     self.training_step_outputs.clear()
    #     self.training_step_targets.clear()

    # def on_validation_epoch_end(self):
        
    #     with open(config.LOGGING_FILE, "a") as f:
    #         pred, label = self.validation_step_outputs[-1], self.validation_step_targets[-1]
    #         f.write(f"Validation \n")
    #         cer = self.cer_fn(pred, label).item()
    #         wer = self.wer_fn(pred, label).item()
    #         f.write(f"Predicted \t: {pred}\n")
    #         f.write(f"Actual \t\t: {label}\n")
    #         f.write(f"CER \t\t: {cer:.4f}\n")
    #         f.write(f"WER \t\t: {wer:.4f}\n\n")
        
    #     # if self.training: TODO: investigate when to log attention plots to wandb
    #     if len(self.self_attn_weights) > 0:
    #         # plot attention weights
    #         plot_attention(
    #             self.self_attn_weights[0], 
    #             show=False, 
    #             pre_fix="val_selfattn", 
    #             folder="val",
    #             epoch=self.current_epoch,
    #             wandb_logging=True
    #         )

    #         plot_attention(
    #             self.cross_attn_weights[0],
    #             kind="cross", 
    #             pre_fix="val_crossattn", 
    #             show=False, 
    #             folder="val",
    #             epoch=self.current_epoch,
    #             wandb_logging=True
    #         )   

    #         plot_attention(
    #             self.cross_attn_tokens_weights[0], 
    #             pre_fix="val_crossattn_tokens", 
    #             show=False, 
    #             folder="val",
    #             epoch=self.current_epoch,
    #             wandb_logging=True
    #         )             
    #     # free memory
    #     self.validation_step_outputs.clear()  
    #     self.validation_step_targets.clear()
    #     self.self_attn_weights.clear()
    #     self.cross_attn_tokens_weights.clear()
    #     self.cross_attn_weights.clear()
    
    def decode_predictions(
            self, 
            preds, 
            tokenizer
    ):
        
        # probs = F.softmax(logits, dim=-1)
        # predicted_tokens = torch.argmax(probs, dim=-1).detach().cpu()

        return tokenizer.decode(preds.tolist())

    
    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=Config.LR
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode="min", 
            factor= 0.1,
            patience= 3
        )
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def update_teacher_forcing_ratio(
            self, 
            epoch:int, 
            start_ratio:float=Config.tf_ratio_start, 
            end_ratio:float=Config.tf_ratio_end, 
            num_epochs:int=Config.EPOCHS
    ):
        self.tf = max(end_ratio, start_ratio - (start_ratio - end_ratio) * (epoch / num_epochs))
        self.log("teacher_forcing_ratio", self.tf)
    
    def on_epoch_end(self):
        self.update_teacher_forcing_ratio(self.current_epoch)


    def log_predictions_and_attention_to_wandb(
        self, 
        src, 
        output, 
        trg, 
        attn_weights, 
        batch_idx
    ):
        B = src.size(0)

        # print(src.shape, trg.shape, output.argmax(dim=-1).shape)

        src_texts       = [self.src_tokenizer.decode(src[idx].cpu().tolist()) for idx in range(B)]
        trg_texts       = [self.tgt_tokenizer.decode(trg[idx].cpu().tolist()) for idx in range(B)]
        output_texts    = [self.decode_predictions(output[idx], tokenizer=self.tgt_tokenizer) for idx in range(B)]

        dist = utils.calc_edit_distance(
            pred=output_texts, 
            trg=trg_texts, 
            tokenizer=self.tgt_tokenizer
        )

        self.log("dist", dist, on_epoch=True, prog_bar=True)

        # Log predictions
        predictions_table = wandb.Table(columns=["Source", "Target", "Prediction"])
        # for i in range(len(src_texts)):
        predictions_table.add_data(src_texts[-1], trg_texts[-1], output_texts[-1])

        wandb.log({"predictions_batch_{}".format(batch_idx): predictions_table})

        # Log attention weights
        # for i, weights in enumerate(attn_weights.cpu()):
        #     attn_figure = wandb.Image(self.plot_attention(weights))
        #     wandb.log({"attention_weights_batch_{}_example_{}".format(batch_idx, i): attn_figure})

    def plot_attention(
            self, 
            attn_weights,
            src_len,
            tgt_len
    ):

        fig, ax = plt.subplots(figsize=(10, 10))
        if src_len is not None:
            sns.heatmap(attn_weights[:tgt_len, :src_len], ax=ax, cmap='GnBu')
        else:
            sns.heatmap(attn_weights, ax=ax, cmap='GnBu')

        plt.xlabel('Source Sequence')
        plt.ylabel('Target Sequence')
        plt.title('Attention Weights')
        plt.show()

        return fig


if __name__=="__main__":

    dm = dataset.build_data_module()
    dm.setup()

    src_vocab_size = dm.src_tokenizer._get_vocab_size()
    tgt_vocab_size = dm.tgt_tokenizer._get_vocab_size()

    print()
    logging.info("Building model")
    model = DyulaTranslator(
        input_dim=src_vocab_size, 
        output_dim=tgt_vocab_size,
        src_tokenizer=dm.src_tokenizer,
        tgt_tokenizer=dm.tgt_tokenizer
        ).to(Config.device)
    
    # print(model)
    summary(model=model)

    source_batch, target_batch, src_lens, tgt_lens = next(iter(dm.train_dataloader()))
    print("Source batch:", source_batch.shape)
    print("Target batch:", target_batch.shape)
    # print("src lens: ", src_lens)
    # print("tgt lens: ", tgt_lens)

    print("> (source); = (target); < (predictions)")

    idx                 = np.random.randint(low=0, high=Config.BATCH_SIZE)
    src                 = source_batch[idx]
    tgt                 = target_batch[idx]
    
    decoded_src         = dm.src_tokenizer.decode(src.tolist())
    print(f"> {decoded_src}")
    decoded_tgt         = dm.tgt_tokenizer.decode(tgt.tolist())
    print(f"= {decoded_tgt}")

    # logits, attn_ws = model(
    #     src=source_batch.to(Config.device), 
    #     trg=target_batch.to(Config.device), 
    #     teacher_forcing_ratio=.8
    # )

    preds, attn_ws = model(
        src=source_batch.to(Config.device), 
        trg=None
    )
    
    print(f"predictions: {preds.shape}, attn_w: {attn_ws.shape}")

    decoded_preds       = model.decode_predictions(
        preds[idx], 
        tokenizer=dm.tgt_tokenizer
    ) 
    # decoded_preds       = dm.tgt_tokenizer.decode(preds[idx].tolist())
    print(f"< {decoded_preds}")

    dist = utils.calc_edit_distance(
        pred=[decoded_preds], 
        trg=[decoded_tgt], 
        tokenizer=dm.tgt_tokenizer
    )
    print(f"Lev. dist: {dist}")

    # print()
    # model.plot_attention(attn_ws[idx], src_len=src_lens[idx], tgt_len=tgt_lens[idx])
