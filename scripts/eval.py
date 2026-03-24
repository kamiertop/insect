from pathlib import Path
from dataclasses import dataclass

import polars as pl
import torch
import typer
from loguru import logger
from torchvision import transforms

from dataset.data_loader import InsectDataset, build_dataloader
from model.model import get_insect_model
from utils import init_logger


@dataclass
class EvalArgs:
    # Data and logging roots.
    data_root: str = "./artifacts/data"
    log_dir: str = "./artifacts/logs"
    split_id: str = ""

    # Evaluation controls.
    checkpoint: str = ""
    batch_size: int = 16
    num_workers: int = 2
    max_steps: int = 0
    save_dir: str = ""


app = typer.Typer(add_completion=False, help="评估脚本，默认读取 artifacts/checkpoints/best.pt")


def init_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_eval_transform() -> transforms.Compose:
    # 评估阶段必须与训练评估分支保持一致，避免分布偏差。
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def align_test_mapping(test_ds: InsectDataset, class_to_idx: dict[str, int]) -> None:
    # 测试集要严格复用训练时的 class_to_idx，确保 label id 语义一致。
    test_labels = set(test_ds.df["label"].to_list())
    train_labels = set(class_to_idx.keys())
    missing = sorted(test_labels - train_labels)
    if missing:
        raise ValueError(f"Test set has labels not seen in training mapping: {missing[:10]}")

    test_ds.class_to_idx = dict(class_to_idx)
    test_ds.idx_to_class = {v: k for k, v in class_to_idx.items()}


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[dict, dict[str, int]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in ckpt:
        raise KeyError("Checkpoint missing key: model_state_dict")
    if "class_to_idx" not in ckpt:
        raise KeyError("Checkpoint missing key: class_to_idx")

    class_to_idx = ckpt["class_to_idx"]
    if not isinstance(class_to_idx, dict) or len(class_to_idx) == 0:
        raise ValueError("Checkpoint class_to_idx is invalid or empty")

    return ckpt, class_to_idx


def resolve_eval_dir(args_save_dir: str, data_root: str, checkpoint_path: Path) -> Path:
    _ = checkpoint_path
    if args_save_dir:
        return Path(args_save_dir)
    # 默认写入固定目录，避免评估产物分散。
    return Path(data_root).parent / "eval"


def resolve_active_split(data_root: str, split_id: str) -> str:
    # Priority: explicit --split-id > marker file > default.
    if split_id.strip():
        return split_id.strip()
    marker = Path(data_root) / "active_split.txt"
    if marker.exists():
        value = marker.read_text(encoding="utf-8").strip()
        if value:
            return value
    return "default"


def resolve_split_path(data_root: str, split_id: str) -> Path:
    return Path(data_root) / "splits" / split_id / "split.csv"


@torch.inference_mode()
def evaluate(
        model: torch.nn.Module,
        dataloader,
        criterion: torch.nn.Module,
        device: torch.device,
        num_classes: int,
        max_steps: int,
) -> tuple[float, float, torch.Tensor]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for step, (images, targets, _) in enumerate(dataloader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        if logits.ndim != 2 or logits.shape[1] != num_classes:
            raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")

        loss = criterion(logits, targets)
        preds = logits.argmax(dim=1)

        bsz = targets.size(0)
        total_loss += loss.item() * bsz
        total_correct += (preds == targets).sum().item()
        total_samples += bsz

        # 逐批累计混淆矩阵，行是真实类别，列是预测类别。
        for t, p in zip(targets.cpu().tolist(), preds.cpu().tolist()):
            confusion[t, p] += 1

        if max_steps > 0 and step >= max_steps:
            break

    avg_loss = safe_div(total_loss, total_samples)
    acc = safe_div(total_correct, total_samples)
    return avg_loss, acc, confusion


def build_metric_tables(confusion: torch.Tensor, idx_to_class: dict[int, str]) -> tuple[
    dict[str, float], list[dict[str, float]], list[list[int]]]:
    num_classes = confusion.shape[0]
    per_class_rows: list[dict[str, float]] = []

    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0

    for idx in range(num_classes):
        tp = float(confusion[idx, idx].item())
        fp = float(confusion[:, idx].sum().item() - tp)
        fn = float(confusion[idx, :].sum().item() - tp)
        support = float(confusion[idx, :].sum().item())

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2.0 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0

        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

        per_class_rows.append({
            "class_idx": idx,
            "label": idx_to_class.get(idx, f"class_{idx}"),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(support),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        })

    macro_precision = safe_div(macro_precision, num_classes)
    macro_recall = safe_div(macro_recall, num_classes)
    macro_f1 = safe_div(macro_f1, num_classes)

    summary = {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }

    cm_rows = confusion.cpu().tolist()
    return summary, per_class_rows, cm_rows


def save_csv_outputs(
        save_dir: Path,
        summary_row: dict[str, float],
        per_class_rows: list[dict[str, float]],
        confusion_rows: list[list[int]],
        idx_to_class: dict[int, str],
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    summary_path = save_dir / "summary.csv"
    per_class_path = save_dir / "per_class.csv"
    confusion_path = save_dir / "confusion_matrix.csv"

    pl.DataFrame([summary_row]).write_csv(summary_path)
    pl.DataFrame(per_class_rows).write_csv(per_class_path)

    # 混淆矩阵列名使用 pred_<label>，方便后续直接查看。
    labels = [idx_to_class[i] for i in range(len(idx_to_class))]
    cm_named_rows: list[dict[str, int | str]] = []
    for i, row in enumerate(confusion_rows):
        item: dict[str, int | str] = {"true_label": labels[i]}
        for j, value in enumerate(row):
            item[f"pred_{labels[j]}"] = int(value)
        cm_named_rows.append(item)
    pl.DataFrame(cm_named_rows).write_csv(confusion_path)

    logger.info(f"saved summary: {summary_path}")
    logger.info(f"saved per-class metrics: {per_class_path}")
    logger.info(f"saved confusion matrix: {confusion_path}")


@app.command()
def main(
        data_root: str = typer.Option("./artifacts/data", help="产物数据根目录"),
        log_dir: str = typer.Option("./artifacts/logs", help="日志输出目录"),
        split_id: str = typer.Option("", help="切分版本 ID，留空则从 active_split.txt 读取"),
        checkpoint: str = typer.Option("", help="模型权重路径，留空默认 artifacts/checkpoints/best.pt"),
        batch_size: int = typer.Option(16, help="评估批大小"),
        num_workers: int = typer.Option(2, help="DataLoader 的 worker 数"),
        max_steps: int = typer.Option(0, help="评估最大步数，0 表示评估完整测试集"),
        save_dir: str = typer.Option("", help="评估输出目录，留空默认 artifacts/eval"),
) -> None:
    args = EvalArgs(
        data_root=data_root,
        log_dir=log_dir,
        split_id=split_id,
        checkpoint=checkpoint,
        batch_size=batch_size,
        num_workers=num_workers,
        max_steps=max_steps,
        save_dir=save_dir,
    )

    init_logger(args.log_dir)

    device = init_device()
    logger.info(f"Device={device}, torch version={torch.__version__}")

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else (Path(args.data_root).parent / "checkpoints" / "best.pt")
    ckpt, class_to_idx = load_checkpoint(checkpoint_path, device)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    eval_dir = resolve_eval_dir(args.save_dir, args.data_root, checkpoint_path)
    resolved_split_id = resolve_active_split(args.data_root, args.split_id)
    resolved_split_path = resolve_split_path(args.data_root, resolved_split_id)

    test_ds = InsectDataset(split_path=resolved_split_path, split="test", transform=build_eval_transform())
    align_test_mapping(test_ds, class_to_idx)

    test_loader = build_dataloader(
        dataset=test_ds,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    num_classes = len(class_to_idx)
    model = get_insect_model(num_classes=num_classes, fine_tune=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    criterion = torch.nn.CrossEntropyLoss()

    test_loss, test_acc, confusion = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        num_classes=num_classes,
        max_steps=args.max_steps,
    )

    summary_extra, per_class_rows, confusion_rows = build_metric_tables(confusion, idx_to_class)
    summary_row = {
        "checkpoint": str(checkpoint_path),
        "resolved_split_id": resolved_split_id,
        "resolved_split_path": str(resolved_split_path),
        "num_classes": num_classes,
        "num_test_samples": int(confusion.sum().item()),
        "test_loss": test_loss,
        "test_acc": test_acc,
        **summary_extra,
    }

    logger.info(
        f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} "
        f"macro_precision={summary_extra['macro_precision']:.4f} "
        f"macro_recall={summary_extra['macro_recall']:.4f} "
        f"macro_f1={summary_extra['macro_f1']:.4f}"
    )

    save_csv_outputs(
        save_dir=eval_dir,
        summary_row=summary_row,
        per_class_rows=per_class_rows,
        confusion_rows=confusion_rows,
        idx_to_class=idx_to_class,
    )


if __name__ == "__main__":
    app()

# 评估模型
