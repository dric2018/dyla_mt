from tqdm import tqdm
from rich.progress import track
from time import sleep


if __name__ == "__main__":
    
    EPOCHS = 2

    for e in range(EPOCHS):
        loop = tqdm(range(50), desc=f"[Epoch {e+1}/{EPOCHS}] Training", colour="blue",leave=True)
        for step in loop:
            sleep(.1)
        

