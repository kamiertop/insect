import torch
from loguru import logger

from utils import init_logger, Config


def train(device: torch.device) -> None:
    return None


def init_device() -> torch.device:
    device: torch.device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    logger.info(f"Device: {device}, torch version: {torch.__version__}")

    return device


if __name__ == '__main__':
    cfg = Config.parse()
    init_logger(cfg.log_dir)
    train(init_device())
