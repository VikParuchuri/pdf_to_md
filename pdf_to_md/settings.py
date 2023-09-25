import os
from typing import Literal, Optional

from dotenv import find_dotenv
from pydantic_settings import BaseSettings
import torch


class Settings(BaseSettings):
    # General settings
    MODEL_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE: int = 1
    BASE_DIR: str = os.path.abspath(os.pardir)
    MODEL_DIR: str = os.path.join(BASE_DIR, "model")

    class Config:
        env_file = find_dotenv("local.env")


settings = Settings()