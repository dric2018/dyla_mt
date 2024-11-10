import os 
import os.path as osp

import torch

class Config:
    # I/O
    STAGE                   = 'debug'
    IS_PRETRAINED           = False
    # Special token IDs
    PAD_TOKEN_ID            = 256      
    EOS_TOKEN_ID            = 257      
    MASK_TOKEN_ID           = 258     # Mask token (optional for span masking)
    SPECIAL_TOKENS          = {
                                PAD_TOKEN_ID: "<pad>", 
                                EOS_TOKEN_ID: "<eos>", 
                                MASK_TOKEN_ID: "<mask>"
                            }
    VOCAB_SIZE              = 259        # 256 bytes + 3 special tokens
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
    NUM_ENCODER_LAYERS      = 12 # 24
    ENCODER_DROPOUT         = .1

    D_MODEL                 = 512 # 1024
    N_HEADS                 = 6 # 16
    ATTN_DROP_RATE          = 0.1
    FF_DROP_RATE            = 0.1
    D_FF                    = 2048 # 4096

    ## Decoder
    NUM_DECODER_LAYERS      = 4 # 8
    DECODER_DROPOUT         = .1
    MAX_OUTPUT              = 128

    SEED                    = 2024
    NUM_WORKERS             = os.cpu_count()
    EMBEDDING_DIM           = 256
    MAX_LENGTH              = 128
    tf_ratio_start          = 0.75
    tf_ratio_end            = 0.35
    BATCH_SIZE              = 4
    LR                      = 1e-4
    EPOCHS                  = 25 if STAGE == 'debug' else 100

    DEVICE                  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

if __name__ == "__main__":
    print(f"Using device: {Config.DEVICE}")
    print(Config.DEVICE.type)