import os 
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
    LOG_DIR                 = osp.join(PROJECT_PATH, "logs")

    # Model
    ## Encoder
    ENCODER_EMBEDDING_DIM   = 256
    ENCODER_HIDDEN_DIM      = 256
    NUM_ENCODER_LAYERS      = 2
    ENCODER_DROPOUT         = .4

    D_MODEL                 = 256

    ## Decoder
    DECODER_EMBEDDING_DIM   = 256
    DECODER_HIDDEN_DIM      = 256
    NUM_DECODER_LAYERS      = 2
    DECODER_DROPOUT         = .4
    MAX_OUTPUT              = 50

    SEED                    = 2024
    NUM_WORKERS             = os.cpu_count()
    EMBEDDING_DIM           = 256
    MAX_LENGTH              = 128
    tf_ratio_start          = 0.75
    tf_ratio_end            = 0.3
    BATCH_SIZE              = 32
    LR                      = 2e-4
    EPOCHS                  = 5 if STAGE == 'debug' else 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

if __name__ == "__main__":
    print(f"Using device: {Config.device}")
    print(Config.device.type)