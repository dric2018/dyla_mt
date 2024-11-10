from config import Config

import lightning as pl

import modules

from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu


import torch

from transformers import AutoModelForSeq2SeqLM, T5ForConditionalGeneration

from torchinfo import summary

import warnings
warnings.filterwarnings("ignore")


# class DyulaTranslator(pl.LightningModule):
#     def __init__(
#             self, 
#             tokenizer, 
#             model_name:str=Config.BACKBONE_MODEL_NAME, 
#             learning_rate:float=Config.LR
#         ):
#         super().__init__()
#         self.translator = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#         self.tokenizer = tokenizer
#         self.learning_rate = learning_rate
#         self.rouge_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

#         # self.validation_step_outputs = []
#         # self.validation_step_targets = []

#     def forward(self, input_ids, attention_mask, labels=None):
#         outputs = self.translator(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#         return outputs.loss, outputs.logits

#     def training_step(self, batch, batch_idx):
#         loss, _ = self.forward(batch['input_ids'], batch['attention_mask'], batch['labels'])
#         self.log("train_loss", loss)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         # Forward pass and loss calculation
#         loss, _ = self.forward(batch['input_ids'], batch['attention_mask'], batch['labels'])
#         self.log("val_loss", loss, prog_bar=True)

#         # Generate predictions
#         outputs = self.translator.generate(
#             batch['input_ids'], 
#             attention_mask=batch['attention_mask'],
#             forced_bos_token_id=self.tokenizer.get_lang_id(Config.OUTPUT_LANG)
#         )
#         preds = [self.tokenizer.decode(g, skip_special_tokens=True) for g in outputs]
#         targets = [self.tokenizer.decode(t, skip_special_tokens=True) for t in batch['labels']]

#         # Collect predictions and targets for BLEU and ROUGE calculation at epoch end
#         outputs = {'preds': preds, 'targets': targets}

#         return {'val_loss': loss, "outputs":outputs}

#     def on_validation_epoch_end(self, outputs):
#         # Aggregate all predictions and targets
#         # all_preds = torch.stack(self.validation_step_outputs)
#         all_preds = sum([x['preds'] for x in outputs], [])

#         # all_targets = torch.stack(self.validation_step_targets)
#         all_targets = sum([x['targets'] for x in outputs], [])

#         # Calculate BLEU score
#         bleu_score = corpus_bleu(all_preds, [all_targets]).score
#         self.log("val_bleu", bleu_score, prog_bar=True)

#         # Calculate ROUGE scores
#         rouge1, rouge2, rougeL = self.compute_rouge_scores(all_preds, all_targets)

#         self.log("val_rouge1", rouge1, prog_bar=True)
#         self.log("val_rouge2", rouge2, prog_bar=True)
#         self.log("val_rougeL", rougeL, prog_bar=True)

#         # self.validation_step_outputs.clear()  # free memory
#         # self.validation_step_targets.clear()  # free memory

#     def compute_royge_scores(self, all_preds, all_targets):
#         # Calculate ROUGE scores
#         rouge1, rouge2, rougeL = 0, 0, 0
#         for pred, target in zip(all_preds, all_targets):
#             scores = self.rouge_scorer.score(target, pred)
#             rouge1 += scores["rouge1"].fmeasure
#             rouge2 += scores["rouge2"].fmeasure
#             rougeL += scores["rougeL"].fmeasure

#         # Average ROUGE scores across the dataset
#         rouge1 /= len(all_preds)
#         rouge2 /= len(all_preds)
#         rougeL /= len(all_preds)

#         return rouge1, rouge2, rougeL
        
#     def configure_optimizers(self):
#         return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


class DyulaTranslator(pl.LightningModule):
    def __init__(
        self, 
        model_name=Config.BACKBONE_MODEL_NAME,
        is_pretrained:bool=True
    ):
        super().__init__()
        if is_pretrained:
            self.translator = T5ForConditionalGeneration.from_pretrained(model_name)
        else:
            self.translator = modules.ByT5Model()

        self.learning_rate = Config.LR

    def forward(self, input_ids, attention_mask, labels=None):
        return self.translator(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        val_loss = outputs.loss
        self.log("val_loss", val_loss)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


if __name__ == '__main__':
    translator = DyulaTranslator(is_pretrained=False)
    print(translator)
    summary(translator)