import os 
import os.path as osp

import torch

class Config:
    # I/O
    STAGE                   = 'debug'
    IS_PRETRAINED           = True
    # Special token IDs
    SOS_TOKEN_ID            = 256
    PAD_TOKEN_ID            = 257      
    EOS_TOKEN_ID            = 258      
    MASK_TOKEN_ID           = 259     # Mask token (optional for span masking)
    SPECIAL_TOKENS          = {
                                SOS_TOKEN_ID: "<sos>",
                                PAD_TOKEN_ID: "<pad>", 
                                EOS_TOKEN_ID: "<eos>", 
                                MASK_TOKEN_ID: "<mask>"
                            }
    VOCAB_SIZE              = 260        # 256 bytes + 4 special tokens
    HF_USERNAME             = "dric2018"
    HF_REPO_NAME            = "dyu-fr-mt"
    BACKBONE_MODEL_NAME     = "google/byt5-small"
    OUTPUT_LANG             = "fr"
    PROJECT_PATH            = "../"
    MODEL_ZOO               = osp.join(PROJECT_PATH, "models")
    DATA_DIR                = osp.join(PROJECT_PATH, "data")
    LOG_DIR                 = osp.join(PROJECT_PATH, "logs")

    # Model
    ## Encoder
    NUM_ENCODER_LAYERS      = 24 # 24
    ENCODER_DROPOUT         = .25

    D_MODEL                 = 1024 # 1024
    N_HEADS                 = 8 # 16
    ATTN_DROP_RATE          = 0.25
    FF_DROP_RATE            = 0.25
    D_FF                    = 2048 # 4096

    ## Decoder
    NUM_DECODER_LAYERS      = 8 # 8
    DECODER_DROPOUT         = .25
    MAX_OUTPUT              = 128

    SEED                    = 2024
    NUM_WORKERS             = os.cpu_count()
    EMBEDDING_DIM           = 512
    MAX_LENGTH              = 32
    tf_ratio_start          = 0.75
    tf_ratio_end            = 0.35
    BATCH_SIZE              = 16
    LR                      = 1e-4
    EPOCHS                  = 2 if STAGE == 'debug' else 100

    DEVICE                  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

if __name__ == "__main__":
    print(f"Using device: {Config.DEVICE}")
    print(Config.DEVICE.type)