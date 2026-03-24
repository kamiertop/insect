import os
from tomllib import load

from pydantic import BaseModel, Field


class EarlyStoppingConfig(BaseModel):
    enabled: bool = True
    patience: int = 5
    min_delta: float = 0.0


class TrainConfig(BaseModel):
    epochs: int = 10
    batch_size: int = 16
    num_workers: int = 2
    lr: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 42
    fine_tune: bool = False
    save_dir: str = "./checkpoints"
    max_train_steps: int = 0
    max_val_steps: int = 0
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)


class Config(BaseModel):
    datasets_path: str
    data_root: str
    log_dir: str
    train: TrainConfig = Field(default_factory=TrainConfig)

    @staticmethod
    def parse(config_path: str = "./config.toml") -> "Config":
        config_path = os.getenv("CONFIG_PATH", config_path)
        with open(config_path, "rb") as f:
            return Config(**load(f))
