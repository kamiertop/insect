import os
from tomllib import load

from pydantic import BaseModel


class Config(BaseModel):
    datasets_path: str
    data_root: str
    log_dir: str

    @staticmethod
    def parse(config_path: str = "./config.toml") -> "Config":
        config_path = os.getenv("CONFIG_PATH", config_path)
        with open(config_path, "rb") as f:
            return Config(**load(f))
