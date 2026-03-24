import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import typer
from loguru import logger
from torchvision import transforms

from dataset.data_loader import InsectDataset, build_dataloader
from model.model import get_insect_model
from utils import init_logger


@dataclass
class TrainArgs:
    data_root: str = "./artifacts/data"
    log_dir: str = "./artifacts/logs"
    split_id: str = ""
    epochs: int = 10
    batch_size: int = 16
    num_workers: int = 2
    lr: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 42
    max_train_steps: int = 0
    max_val_steps: int = 0
    fine_tune: bool = False
    early_stop_enabled: bool = True
    early_stop_patience: int = 5
    early_stop_min_delta: float = 0.0


app = typer.Typer(add_completion=False, help="训练脚本，执行前需要先生成标签切分文件")


def active_split_marker_path(data_root: str) -> Path:
    return Path(data_root) / "active_split.txt"


def resolve_active_split(data_root: str, split_id: str) -> str:
    # Priority: explicit --split-id > marker file > default.
    if split_id.strip():
        return split_id.strip()
    marker = active_split_marker_path(data_root)
    if marker.exists():
        value = marker.read_text(encoding="utf-8").strip()
        if value:
            return value
    return "default"


def resolve_split_path(data_root: str, split_id: str) -> Path:
    return Path(data_root) / "splits" / split_id / "split.csv"


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


def train(args: TrainArgs, device: torch.device, split_path: Path) -> None:
    train_ds = InsectDataset(split_path=split_path, split="train", transform=build_train_transform())
    val_ds = InsectDataset(split_path=split_path, split="val", transform=build_eval_transform())
    align_val_mapping(train_ds, val_ds)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("split file has empty train or val split, cannot run training")

    if train_ds.num_classes < 2:
        raise ValueError(f"Need at least 2 classes to train, got {train_ds.num_classes}")

    train_loader = build_dataloader(
        dataset=train_ds,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    logger.info("build train dataloader done")
    val_loader = build_dataloader(
        dataset=val_ds,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    logger.info("build val dataloader done")
    model = get_insect_model(num_classes=train_ds.num_classes, fine_tune=args.fine_tune).to(device)
    logger.info("build insect model done")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # 训练权重固定输出到统一目录，便于集中管理。
    save_dir = Path(args.data_root).parent / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "best.pt"
    last_path = save_dir / "last.pt"

    logger.info(f"best model path: {best_path}")
    logger.info(f"last model path: {last_path}")

    best_val_acc = -1.0
    no_improve_epochs = 0
    total_elapsed_sec = 0.0
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_acc = run_train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_classes=train_ds.num_classes,
            max_steps=args.max_train_steps,
        )
        val_loss, val_acc = run_val_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=train_ds.num_classes,
            max_steps=args.max_val_steps,
        )
        epoch_elapsed_sec = time.perf_counter() - epoch_start
        total_elapsed_sec += epoch_elapsed_sec

        logger.info(
            f"epoch={epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"epoch_time={epoch_elapsed_sec:.2f}s total_time={total_elapsed_sec:.2f}s"
        )

        improved = val_acc > (best_val_acc + args.early_stop_min_delta)
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

        if args.early_stop_enabled:
            logger.info(
                f"early_stopping status: no_improve={no_improve_epochs}/{args.early_stop_patience} "
                f"min_delta={args.early_stop_min_delta}"
            )
            if no_improve_epochs >= args.early_stop_patience:
                logger.warning(
                    f"Early stopping triggered at epoch {epoch}: "
                    f"no improvement for {no_improve_epochs} epochs"
                )
                break

    logger.info(f"Training done. last={last_path} best={best_path} best_val_acc={best_val_acc:.4f}")


@app.command()
def main(
        data_root: str = typer.Option("./artifacts/data", help="产物数据根目录"),
        log_dir: str = typer.Option("./artifacts/logs", help="日志输出目录"),
        split_id: str = typer.Option("", help="切分版本 ID，留空则从 active_split.txt 读取"),
        epochs: int = typer.Option(10, help="训练总轮数"),
        batch_size: int = typer.Option(16, help="训练批大小"),
        num_workers: int = typer.Option(2, help="DataLoader 的 worker 数"),
        lr: float = typer.Option(1e-4, help="优化器学习率"),
        weight_decay: float = typer.Option(1e-4, help="权重衰减系数"),
        seed: int = typer.Option(42, help="随机种子"),
        max_train_steps: int = typer.Option(0, help="单轮训练最多步数，0 表示跑完整个 epoch"),
        max_val_steps: int = typer.Option(0, help="单轮验证最多步数，0 表示跑完整个验证集"),
        fine_tune: bool = typer.Option(False, "--fine-tune/--no-fine-tune", help="是否启用微调（解冻部分骨干网络）"),
        early_stop_enabled: bool = typer.Option(True, "--early-stop-enabled/--no-early-stop-enabled", help="是否启用早停"),
        early_stop_patience: int = typer.Option(5, help="早停容忍轮数（连续多少轮无提升后停止）"),
        early_stop_min_delta: float = typer.Option(0.0, help="判定为提升所需的最小增量"),
) -> None:
    args = TrainArgs(
        data_root=data_root,
        log_dir=log_dir,
        split_id=split_id,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        max_train_steps=max_train_steps,
        max_val_steps=max_val_steps,
        fine_tune=fine_tune,
        early_stop_enabled=early_stop_enabled,
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
    )

    init_logger(args.log_dir)

    resolved_split_id = resolve_active_split(args.data_root, args.split_id)
    resolved_split_path = resolve_split_path(args.data_root, resolved_split_id)
    checkpoint_dir = Path(args.data_root).parent / "checkpoints"
    eval_dir = Path(args.data_root).parent / "eval"
    logger.info(f"checkpoint_dir={checkpoint_dir}")
    logger.info(f"eval_dir_hint={eval_dir}")

    set_seed(args.seed)
    device = init_device()
    logger.info(f"Device={device}, torch version={torch.__version__}")
    logger.info(
        "Training args: "
        f"epochs={args.epochs} batch_size={args.batch_size} num_workers={args.num_workers} "
        f"lr={args.lr} weight_decay={args.weight_decay} fine_tune={args.fine_tune} "
        f"early_stop={args.early_stop_enabled} patience={args.early_stop_patience}"
    )
    logger.info(f"Using split: id={resolved_split_id} path={resolved_split_path}")

    train(args=args, device=device, split_path=resolved_split_path)

    return None


if __name__ == "__main__":
    app()
