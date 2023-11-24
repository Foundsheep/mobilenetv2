import torch
import pytz

ROOT = "./"
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCHS = 40
WIDTH_MULTIPLIER = 1
RESOLUTION_MULTIPLIER = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEOUL_TZ = pytz.timezone("Asia/Seoul")