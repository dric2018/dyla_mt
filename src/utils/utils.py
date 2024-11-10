from config import Config

from datetime import datetime

import logging
logging.basicConfig(level='INFO')
import Levenshtein

import os

import modules

import numpy as np
import pandas as pd

import random

from tqdm import tqdm
import torch

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

def linear_tf_scheduler(
    epoch, 
    start_ratio=Config.tf_ratio_start, 
    end_ratio=Config.tf_ratio_end, 
    num_epochs=Config.EPOCHS
):
    return max(end_ratio, start_ratio - (start_ratio - end_ratio) * (epoch / num_epochs))


def generate_experiment_name():
    # Get the current date in YYYY-MM-DD format
    date_str = datetime.now().strftime("%Y-%m-%d")
    
    # List of adjectives and nouns for unique experiment names
    adjectives = ["brave", "bold", "bright", "clever", "curious", "mighty", "swift", "wise", "gentle", "fierce"]
    nouns = ["falcon", "tiger", "lion", "eagle", "phoenix", "orca", "wolf", "panther", "dragon", "hawk"]

    # Randomly select an adjective and a noun
    adjective = random.choice(adjectives)
    noun = random.choice(nouns)
    
    # Combine date, adjective, and noun to create a unique name
    experiment_name = f"{date_str}_{adjective}_{noun}"
    
    return experiment_name


def train_fn(
        model, 
        data_loader, 
        optimizer,
        device=Config.DEVICE, 
        teacher_forcing_ratio=Config.tf_ratio_start
):
    model.train()
    epoch_loss = 0
    
    loop = tqdm(data_loader, leave=False, desc="Training", colour="blue")
    
    for batch in loop:
        src_batch, trg_batch, _, _ = batch
        src_batch, trg_batch = src_batch.to(device), trg_batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output, attn_ws = model(src=src_batch, trg=trg_batch, teacher_forcing_ratio=teacher_forcing_ratio)
        # print(f"out: {output.shape}")
        
        output = output[:, 1:].reshape(-1, output.shape[-1])
        trg_batch = trg_batch[:, 1:].reshape(-1)
        
        loss = model.criterion(output, trg_batch)
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
        loop.set_postfix_str(f"Loss: {loss.item():.4f}")
    
    return epoch_loss / len(data_loader)


def valid_fn(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    
    # Initialize the progress bar
    loop = tqdm(data_loader, leave=False)
    
    with torch.no_grad():
        for batch in loop:
            src_batch, trg_batch = batch
            src_batch, trg_batch = src_batch.to(device), trg_batch.to(device)
            
            # Forward pass
            output = model(src_batch, trg_batch, teacher_forcing_ratio=0)  # No teacher forcing during validation
            
            # Reshape for loss calculation
            output = output[:, 1:].reshape(-1, output.shape[-1])
            trg_batch = trg_batch[:, 1:].reshape(-1)
            
            # Calculate loss
            loss = criterion(output, trg_batch)
            
            # Accumulate the loss
            epoch_loss += loss.item()
            
            # Update the progress bar
            loop.set_description(f"Validation Loss: {loss.item():.4f}")
    
    return epoch_loss / len(data_loader)


def experiment(model, train_loader, valid_loader, optimizer, criterion, device, num_epochs, save_path, lr_scheduler=None, tf_scheduler=None):
    best_valid_loss = float('inf')
    teacher_forcing_ratio = 1.0  # Start with full teacher forcing

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        
        # Training step
        train_loss = train_fn(model, train_loader, criterion, optimizer, device, teacher_forcing_ratio)
        
        # Validation step
        valid_loss = valid_fn(model, valid_loader, criterion, device)
        
        # Print epoch summary
        print(f'\nTrain Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}')
        
        # Save the model if validation loss improves
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), save_path)
            print(f'Model saved with validation loss: {valid_loss:.4f}')
        
        # Step the learning rate scheduler if provided
        if lr_scheduler is not None:
            lr_scheduler.step()
            print(f'Learning Rate after step: {lr_scheduler.get_last_lr()}')
        
        # Adjust the teacher forcing ratio if scheduler provided
        if tf_scheduler is not None:
            teacher_forcing_ratio = tf_scheduler(epoch + 1)
            print(f'Teacher Forcing Ratio after step: {teacher_forcing_ratio:.4f}')
        
        print('-' * 50)


def calc_edit_distance(pred, trg, tokenizer):

    distance = 0 
    batch_size = len(pred) if isinstance(pred, list) else trg.shape[0] 

    if isinstance(pred, list):
        for idx in range(batch_size):
            distance += Levenshtein.distance(trg[idx], pred[idx]) 
    else:
        for i in range(batch_size): 
            decoded_trg = tokenizer.decode(trg[i].tolist())
            decoded_pred = tokenizer.decode(torch.argmax(pred[i],axis=-1).tolist())
            distance += Levenshtein.distance(decoded_trg, decoded_pred) 
            
    distance /= batch_size 

    return distance

def fetch_sample_from_batch(
    batch, 
    batch_size:int, 
    random:bool=False,
    idx:int=0
):
    
    if random:
        idx = np.random.randint(batch_size)
    else:
        assert idx < batch_size, f"Value of idx ({idx}) is higher than batch size ({batch_size})"
        idx = idx
        
    return batch[idx]

def load_checkpoint(
        hparams,
        model_name:str="dyula_mt", 
        device:str=Config.DEVICE,
    ):
    
    logging.info("Loading model from checkpoint...")
    logging.info("Creating instance of RTCRAM...")
    
    model = modules.DyulaTranslator(
    **hparams
    ).to(device)
    
    logging.info("Preparing checkpoint...")
    CKPT_PATH = os.path.join(Config.MODEL_PATH, model_name+".bin")
    
    logging.info("loading model state dict...")
    try:
        ckpt = torch.load(CKPT_PATH)  
        model.load_state_dict(ckpt["model_state_dict"])
    except FileNotFoundError:
        logging.error(f"Could not find model checkpoint in {Config.MODEL_PATH}...trying other locations")
        ckpt = torch.load(os.path.join("models", model_name+".bin"))  
        model.load_state_dict(ckpt["model_state_dict"])

    logging.info("Loading model from checkpoint...Complete!")
    
    return model

def predict(model, inp, debug, device):

    # greedy decoding

    # decode predictions
    
    # return preds and attention weigths
    pass

def inference_step(
    test_loader, 
    debug:bool=False, 
    mode:str="inference",
    device:str=Config.DEVICE
):
    
    """
        Execute inference procedure in 2 modes
        
        mode: (str) One of 
            inference: generate predictions using trained model
            eval: Evaluate model by comparing predictions to ground truth
            
        test_loader: (DataLoader) Data loader to be used for testing
    """
    
    model = load_checkpoint(model_name="RTCRAM")
    
    output = {
        "self_attn_ws"          : [], 
        "cross_attn_ws"         : [],
        "preds"         : [],
        "labels" : []
    }
    logging.info("Running inference now...")

    
    test_progress = tqdm(range(len(test_loader)), desc="Running inference")
    
    for batch_num in test_progress:
        batch = next(iter(test_loader))
        
        for sample_id in tqdm(range(Config.TEST_BATCH_SIZE), leave=False, desc="Generating motor commands"):
        
            inp = fetch_sample_from_batch(
                batch, 
                batch_size=batch["in_state"].shape[0],
                random=False,
                idx=sample_id
            )

            preds, self_attn_ws, cross_attn_ws = predict(model, inp, debug, device)

            output["self_attn_ws"].append(self_attn_ws)
            output["cross_attn_ws"].append(cross_attn_ws)
            output["preds"].append(preds[0])

            if mode == "eval":
                labels = inp["labels"].to(Config.DEVICE)
                label = model.decode_predictions(
                    predicted_ids=labels
                )
            output["labels"].append(label[0])
        
        # break
        
    if mode == "eval":
        test_dist = calc_edit_distance(
            predictions=output["preds"], 
            y=output["labels"], 
            batch=True
        )        
        
        inference_results = pd.DataFrame({
            "prediction": output["preds"],
            "label": output["labels"],
            "correct": [float(p==l) for p,l in zip(output["preds"], output["labels"])],
            "distance": [calc_edit_distance(p, l, batch=False) for p,l in zip(output["preds"], output["labels"])]
        })
        
        success_rate = 100*inference_results.correct.mean()
        
        print(f"**** Evaluatiion Report *****")
        print(f"> Test Lev. distance\t: {test_dist:.4f}")
        print(f"> Success Rate\t\t: {success_rate:.4f}%")
        print(f"**** Evaluatiion Report *****")
        return inference_results
    else:
        return output