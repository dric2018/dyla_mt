

from nltk.translate.bleu_score import sentence_bleu

def evaluate(model, dataloader, tokenizer):
    model.eval()
    bleu_scores = []
    for batch in dataloader:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        
        outputs = model.generate(input_ids)
        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        targets = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        for pred, target in zip(preds, targets):
            bleu_scores.append(sentence_bleu([target.split()], pred.split()))
    
    return sum(bleu_scores) / len(bleu_scores)

# Evaluate on validation set
# val_loader = data_module.val_dataloader()
# bleu_score = evaluate(model, val_loader, data_module.tokenizer)
# print(f"Validation BLEU Score: {bleu_score:.2f}")
