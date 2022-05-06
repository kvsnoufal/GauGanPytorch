class Config:
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256
    INPUT_FILES = "../data/facades_data/*.jpg"
    BATCH_SIZE = 8
    LATENT_DIM = 256
    DEVICE = "cuda"
    NUM_CLASSES = 12
    TEST_SPLIT = 0.2
    EPOCHS = 200
    OUTPUT_FOLDER = "output_flickr"
    SAVE_MODEL_FOLDER = "model"
    SAVE_LOGS_FOLDER = "logs"
    RUN_ID = "3_CONTD"
    SAVE_EVERY = 2
    EVAL_EVERY = 1

    ENCODER_WEIGHTS = "output_flickr/model/2_CONTD/enc_109.pt"
    GENERATOR_WEIGHTS = "output_flickr/model/2_CONTD/gen_109.pt"
    DISCRIMINATOR_WEIGHTS = "output_flickr/model/2_CONTD/disc_109.pt"

    ENCODER_WEIGHTS_EVAL = "output_flickr/model/3_CONTD/enc_37.pt"
    GENERATOR_WEIGHTS_EVAL = "output_flickr/model/3_CONTD/gen_37.pt"
    
    
