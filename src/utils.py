from config import Config
from tqdm import tqdm

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


def train_fn(model, data_loader, criterion, optimizer, device, teacher_forcing_ratio):
    model.train()
    epoch_loss = 0
    
    loop = tqdm(data_loader, leave=False)
    
    for batch in loop:
        src_batch, trg_batch = batch
        src_batch, trg_batch = src_batch.to(device), trg_batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(src_batch, trg_batch, teacher_forcing_ratio=teacher_forcing_ratio)
        
        output = output[:, 1:].reshape(-1, output.shape[-1])
        trg_batch = trg_batch[:, 1:].reshape(-1)
        
        loss = criterion(output, trg_batch)
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
        loop.set_description(f"Training Loss: {loss.item():.4f}")
    
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



