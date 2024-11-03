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
    BACKBONE_MODEL_NAME     = "facebook/m2m100_418M" #"xlm-roberta-base"
    OUTPUT_LANG             = "fr"
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
    MAX_LENGTH              = 64
    tf_ratio_start          = 0.75
    tf_ratio_end            = 0.35
    BATCH_SIZE              = 8
    LR                      = 1e-4
    EPOCHS                  = 5 if STAGE == 'debug' else 100

    DEVICE                  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

if __name__ == "__main__":
    print(f"Using device: {Config.DEVICE}")
    print(Config.DEVICE.type)