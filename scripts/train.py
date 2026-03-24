import argparse
import random
import time
from pathlib import Path

import torch
from loguru import logger
from torchvision import transforms

from data.data_loader import InsectDataset, build_dataloader
from model.model import get_insect_model
from utils import Config, init_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train insect image classifier")
    parser.add_argument("--config", type=str, default="./config.toml", help="Path to config.toml")
    return parser.parse_args()


def init_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    logger.info(f"Setting seed: {seed}")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def align_val_mapping(train_ds: InsectDataset, val_ds: InsectDataset) -> None:
    # 评估时必须沿用训练集标签映射，避免 label id 语义错位。
    train_map = dict(train_ds.class_to_idx)
    missing = sorted(set(val_ds.df["label"].to_list()) - set(train_map.keys()))
    if missing:
        raise ValueError(f"Validation set has labels not seen in train set: {missing[:10]}")
    val_ds.class_to_idx = train_map
    val_ds.idx_to_class = {v: k for k, v in train_map.items()}


def run_train_epoch(
        model: torch.nn.Module,
        dataloader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        num_classes: int,
        max_steps: int,
) -> tuple[float, float]:
    """
    Args:
        model:
        dataloader:
        criterion:
        optimizer:
        device: cpu or cuda
        num_classes: 分类数量
        max_steps:

    Returns:

    """
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for step, (images, targets, _) in enumerate(dataloader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        if logits.ndim != 2 or logits.shape[1] != num_classes:
            raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")

        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        bsz = targets.size(0)
        running_loss += loss.item() * bsz
        running_correct += (logits.argmax(dim=1) == targets).sum().item()
        running_total += bsz

        if max_steps > 0 and step >= max_steps:
            break

    return running_loss / max(running_total, 1), running_correct / max(running_total, 1)


@torch.inference_mode()
def run_val_epoch(
        model: torch.nn.Module,
        dataloader,
        criterion: torch.nn.Module,
        device: torch.device,
        num_classes: int,
        max_steps: int,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for step, (images, targets, _) in enumerate(dataloader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        if logits.ndim != 2 or logits.shape[1] != num_classes:
            raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")

        loss = criterion(logits, targets)
        bsz = targets.size(0)
        running_loss += loss.item() * bsz
        running_correct += (logits.argmax(dim=1) == targets).sum().item()
        running_total += bsz

        if max_steps > 0 and step >= max_steps:
            break

    return running_loss / max(running_total, 1), running_correct / max(running_total, 1)


def save_checkpoint(
        path: Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        best_val_acc: float,
        class_to_idx: dict[str, int],
) -> None:
    """
    保存训练检查点，包括模型状态、优化器状态、当前轮次、最佳验证准确率和类别映射。
    Args:
        path: checkpoint 保存路径
        model:  model.state_dict() 包含模型参数和缓冲区，足以恢复模型状态
        optimizer: optimizer 状态也一并保存，方便后续恢复训练
        epoch: 轮次
        best_val_acc: 当前最佳验证准确率
        class_to_idx: 类别名称到 id 的映射，评估时需要保证与训练时一致
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "class_to_idx": class_to_idx,
        },
        path,
    )


def train(cfg: Config, device: torch.device) -> None:
    train_cfg = cfg.train
    train_ds = InsectDataset(config=cfg, split="train", transform=build_train_transform())
    val_ds = InsectDataset(config=cfg, split="val", transform=build_eval_transform())
    align_val_mapping(train_ds, val_ds)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("train.csv or val.csv is empty, cannot run training")

    if train_ds.num_classes < 2:
        raise ValueError(f"Need at least 2 classes to train, got {train_ds.num_classes}")

    train_loader = build_dataloader(
        dataset=train_ds,
        shuffle=True,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
    )
    logger.info("build train dataloader done")
    val_loader = build_dataloader(
        dataset=val_ds,
        shuffle=False,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
    )
    logger.info("build val dataloader done")
    model = get_insect_model(num_classes=train_ds.num_classes, fine_tune=train_cfg.fine_tune).to(device)
    logger.info("build insect model done")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    save_dir = Path(train_cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "best.pt"
    last_path = save_dir / "last.pt"

    logger.info(f"best model path: {best_path}")
    logger.info(f"last model path: {last_path}")

    best_val_acc = -1.0
    no_improve_epochs = 0
    total_elapsed_sec = 0.0
    for epoch in range(1, train_cfg.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_acc = run_train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_classes=train_ds.num_classes,
            max_steps=train_cfg.max_train_steps,
        )
        val_loss, val_acc = run_val_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=train_ds.num_classes,
            max_steps=train_cfg.max_val_steps,
        )
        epoch_elapsed_sec = time.perf_counter() - epoch_start
        total_elapsed_sec += epoch_elapsed_sec

        logger.info(
            f"epoch={epoch}/{train_cfg.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"epoch_time={epoch_elapsed_sec:.2f}s total_time={total_elapsed_sec:.2f}s"
        )

        improved = val_acc > (best_val_acc + train_cfg.early_stopping.min_delta)
        if improved:
            best_val_acc = val_acc
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        save_checkpoint(
            path=last_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_acc=best_val_acc,
            class_to_idx=train_ds.class_to_idx,
        )

        if improved:
            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_acc=best_val_acc,
                class_to_idx=train_ds.class_to_idx,
            )
            logger.success(f"New best model saved: {best_path} (val_acc={best_val_acc:.4f})")

        if train_cfg.early_stopping.enabled:
            logger.info(
                f"early_stopping status: no_improve={no_improve_epochs}/{train_cfg.early_stopping.patience} "
                f"min_delta={train_cfg.early_stopping.min_delta}"
            )
            if no_improve_epochs >= train_cfg.early_stopping.patience:
                logger.warning(
                    f"Early stopping triggered at epoch {epoch}: "
                    f"no improvement for {no_improve_epochs} epochs"
                )
                break

    logger.info(f"Training done. last={last_path} best={best_path} best_val_acc={best_val_acc:.4f}")


def main() -> None:
    args = parse_args()
    cfg = Config.parse(args.config)
    init_logger(cfg.log_dir)
    set_seed(cfg.train.seed)
    device = init_device()
    logger.info(f"Device={device}, torch version={torch.__version__}")
    logger.info(f"Training config: {cfg.train.model_dump()}")

    train(cfg=cfg, device=device)

    return None


if __name__ == "__main__":
    main()
