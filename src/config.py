
import os.path as osp
import torch

class Config:

    # I/O
    STAGE                   = 'debug'
    unk_token               = "[UNK]"
    pad_token               = "[PAD]"
    sos_token               = "[SOS]"
    eos_token               = "[EOS]"
    SPECIAL_TOKENS          = [
                                pad_token,
                                sos_token,
                                eos_token,
                                unk_token
                            ]
    HF_USERNAME             = "dric2018"
    HF_REPO_NAME            = "dyu-fr-mt"
    PROJECT_PATH            = "../"
    MODEL_ZOO               = osp.join(PROJECT_PATH, "models")
    DATA_DIR                = osp.join(PROJECT_PATH, "data")

    # Model
    SEED                    = 2024
    MAX_LENGTH              = 128
    teacher_forcing_ratio   = 0.5
    BATCH_SIZE              = 32
    LR                      = 1e-3
    EPOCHS                  = 25 if STAGE == 'debug' else 150

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

if __name__ == "__main__":
    print(f"Using device: {Config.device}")