import torch
import torch.nn as nn
import wandb

from config import Config


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
    
    def forward(self, x):
        return self.embedding(x)
    
class Encoder(nn.Module):
    def __init__(self, emb_dim, hid_dim, n_layers, dropout):
        super(Encoder, self).__init__()
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src_emb):
        outputs, (hidden, cell) = self.rnn(self.dropout(src_emb))
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, emb_dim, hid_dim, output_dim, n_layers, dropout):
        super(Decoder, self).__init__()
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, trg_emb, hidden, cell):
        outputs, (hidden, cell) = self.rnn(self.dropout(trg_emb), (hidden, cell))
        predictions = self.fc_out(outputs)
        return predictions, hidden, cell


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = d_model ** 0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # Compute attention scores
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute context vectors
        output = torch.matmul(attn_weights, value)
        return output, attn_weights


class Seq2SeqWithAttention(nn.Module):
    def __init__(
        self, 
        encoder, 
        decoder, 
        attention, 
        src_emb_layer, 
        trg_emb_layer, 
        trg_sos_idx, 
        trg_eos_idx, 
        max_trg_len=50
    ):
        super(Seq2SeqWithAttention, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.attention = ScaledDotProductAttention()
        self.src_emb_layer = EmbeddingLayer()
        self.trg_emb_layer = EmbeddingLayer()
        self.trg_sos_idx = trg_sos_idx
        self.trg_eos_idx = trg_eos_idx
        self.max_trg_len = max_trg_len

    def forward(self, src, trg=None, teacher_forcing_ratio=1.0):
        src_emb = self.src_emb_layer(src)
        hidden, cell = self.encoder(src_emb)
        
        trg_len = trg.size(1) if trg is not None else self.max_trg_len
        batch_size = src.size(0)
        output_dim = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch_size, trg_len, output_dim).to(src.device)
        
        decoder_input = torch.tensor([[self.trg_sos_idx]] * batch_size).to(src.device)
        decoder_input = self.trg_emb_layer(decoder_input)
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            output, attn_weights = self.attention(query=output, key=src_emb, value=src_emb)
            outputs[:, t] = output.squeeze(1)
            
            if trg is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = self.trg_emb_layer(trg[:, t].unsqueeze(1))
            else:
                top1 = output.argmax(2)
                decoder_input = self.trg_emb_layer(top1)
            
            if trg is None and (top1 == self.trg_eos_idx).all():
                break

        return outputs, attn_weights



class DyulaTranslator(pl.LightningModule):
    def __init__(
        self, 
        encoder, 
        decoder, 
        attention, 
        src_emb_layer, 
        trg_emb_layer, 
        trg_sos_idx, 
        trg_eos_idx, 
        max_trg_len=50, 
        learning_rate=0.001
    ):
        super(DyulaTranslator, self).__init__()
        # Initialization as before
        self.attention = attention
        # ...

    def validation_step(self, batch, batch_idx):
        src, trg = batch
        output, attn_weights = self(src, trg, teacher_forcing_ratio=0)

        output = output[:, 1:].reshape(-1, output.shape[-1])
        trg = trg[:, 1:].reshape(-1)

        # Calculate loss
        loss = self.criterion(output, trg)
        self.log("val_loss", loss)

        # Log predictions and attention weights to W&B
        if batch_idx % 100 == 0:  # Adjust frequency as needed
            self.log_predictions_and_attention_to_wandb(src, output, trg, attn_weights, batch_idx)

        return loss

    def log_predictions_and_attention_to_wandb(self, src, output, trg, attn_weights, batch_idx):
        src_texts = [self.src_tokenizer.decode(s) for s in src.cpu()]
        trg_texts = [self.trg_tokenizer.decode(t) for t in trg.cpu()]
        output_texts = [self.trg_tokenizer.decode(o.argmax(dim=-1)) for o in output.cpu()]

        # Log predictions
        predictions_table = wandb.Table(columns=["Source", "Target", "Prediction"])
        for i in range(len(src_texts)):
            predictions_table.add_data(src_texts[i], trg_texts[i], output_texts[i])

        wandb.log({"predictions_batch_{}".format(batch_idx): predictions_table})

        # Log attention weights
        for i, weights in enumerate(attn_weights.cpu()):
            attn_figure = wandb.Image(self.plot_attention(weights))
            wandb.log({"attention_weights_batch_{}_example_{}".format(batch_idx, i): attn_figure})

    def plot_attention(self, attn_weights):
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(attn_weights, ax=ax, cmap='viridis')
        plt.xlabel('Source Sequence')
        plt.ylabel('Target Sequence')
        plt.title('Attention Weights')
        plt.show()
        return fig
