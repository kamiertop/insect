import random
from pathlib import Path

import torch
import typer
from loguru import logger
from torchvision import transforms

from dataset.data_loader import InsectDataset, build_dataloader
from model.model import get_insect_model
from utils import init_logger

app = typer.Typer(add_completion=False, help="单轮 smoke 训练脚本")


def init_device() -> torch.device:
	if torch.cuda.is_available():
		return torch.device("cuda")
	return torch.device("cpu")


def set_seed(seed: int) -> None:
	random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def build_train_transform() -> transforms.Compose:
	return transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])


def train_one_epoch(
	device: torch.device,
	split_path: str,
	lr: float,
	batch_size: int,
	num_workers: int,
	fine_tune: bool,
	max_steps: int,
) -> None:
	dataset = InsectDataset(split_path=split_path, split="train", transform=build_train_transform())
	if len(dataset) == 0:
		raise ValueError("train split is empty in split.csv, cannot run training")

	num_classes = len(dataset.class_to_idx)
	if num_classes < 2:
		raise ValueError(f"Need at least 2 classes to train, got {num_classes}")

	dataloader = build_dataloader(
		dataset=dataset,
		shuffle=True,
		batch_size=batch_size,
		num_workers=num_workers,
	)

	model = get_insect_model(num_classes=num_classes, fine_tune=fine_tune).to(device)
	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.AdamW(
		[p for p in model.parameters() if p.requires_grad],
		lr=lr,
		weight_decay=1e-4,
	)

	model.train()
	running_loss = 0.0
	running_correct = 0
	running_total = 0

	logger.info(
		f"Start 1 epoch | device={device} samples={len(dataset)} classes={num_classes} "
		f"batch_size={batch_size} steps={len(dataloader)}"
	)

	for step, (images, targets, _) in enumerate(dataloader, start=1):
		images = images.to(device, non_blocking=True)
		targets = targets.to(device, non_blocking=True)

		optimizer.zero_grad()
		logits = model(images)
		if logits.ndim != 2 or logits.shape[1] != num_classes:
			raise RuntimeError(
				f"Unexpected logits shape: {tuple(logits.shape)}, expected [N, {num_classes}]"
			)

		loss = criterion(logits, targets)
		loss.backward()
		optimizer.step()

		batch_size_now = targets.size(0)
		running_loss += loss.item() * batch_size_now
		preds = logits.argmax(dim=1)
		running_correct += (preds == targets).sum().item()
		running_total += batch_size_now

		if step % 20 == 0 or step == len(dataloader):
			avg_loss = running_loss / max(running_total, 1)
			avg_acc = running_correct / max(running_total, 1)
			logger.info(f"step={step}/{len(dataloader)} loss={avg_loss:.4f} acc={avg_acc:.4f}")

		if max_steps > 0 and step >= max_steps:
			logger.warning(f"Early stop for smoke run: reached max_steps={max_steps}")
			break

	epoch_loss = running_loss / max(running_total, 1)
	epoch_acc = running_correct / max(running_total, 1)
	logger.success(f"Epoch done | loss={epoch_loss:.4f} acc={epoch_acc:.4f} seen={running_total}")


@app.command()
def main(
		data_root: str = typer.Option("./artifacts/data", help="产物数据根目录"),
		log_dir: str = typer.Option("./artifacts/logs", help="日志输出目录"),
		split_id: str = typer.Option("", help="切分版本 ID，留空则从 active_split.txt 读取"),
		seed: int = typer.Option(42, help="随机种子"),
		lr: float = typer.Option(1e-4, help="学习率"),
		batch_size: int = typer.Option(16, help="批大小"),
		num_workers: int = typer.Option(2, help="DataLoader 的 worker 数"),
		fine_tune: bool = typer.Option(False, "--fine-tune/--no-fine-tune", help="是否启用微调"),
		max_steps: int = typer.Option(0, help="最多训练步数，0 表示完整 epoch"),
) -> None:

	# Priority: explicit --split-id > marker file > default.
	marker = Path(data_root) / "active_split.txt"
	resolved_split_id = split_id.strip() or (marker.read_text(encoding="utf-8").strip() if marker.exists() else "default")
	if not resolved_split_id:
		resolved_split_id = "default"
	resolved_split_path = Path(data_root) / "splits" / resolved_split_id / "split.csv"

	init_logger(log_dir)
	set_seed(seed)
	device = init_device()
	logger.info(f"Torch={torch.__version__} | device={device}")

	train_one_epoch(
		device=device,
		split_path=str(resolved_split_path),
		lr=lr,
		batch_size=batch_size,
		num_workers=num_workers,
		fine_tune=fine_tune,
		max_steps=max_steps,
	)


if __name__ == "__main__":
	app()

