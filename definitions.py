"""
Definitions we want to share with other files
"""
from pathlib import Path
import dotenv
import os

dotenv.load_dotenv()

ROOT_DIR = str(Path(__file__).resolve().parents[0])

DATA_DIR = os.getenv("DATA_DIR")
MODEL_DIR = os.getenv("MODEL_DIR")
TENSORBOARD_DIR = os.getenv("TENSORBOARD_DIR")
# TODO: add handeling of aws env
