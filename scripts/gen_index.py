import json
import random
from datetime import datetime
from pathlib import Path

import polars as pl
import typer
from loguru import logger

from utils import init_logger


app = typer.Typer(add_completion=False, help="从 label.csv 生成 split 索引文件")


class DivideRatio:
    """
    DivideRatio 用于表示训练集、验证集、测试集的比例，并提供根据总组数计算每个集合的组数的方法。
     - train: 训练集比例
     - val: 验证集比例
     - test: 测试集比例
     注意：train + val + test 应该等于 1.0
    """

    def __init__(self, train: float, val: float, test: float) -> None:
        self.train = train
        self.val = val
        self.test = test

    def _sum(self) -> float:
        return self.train + self.val + self.test

    def divide(self, n_groups: int) -> tuple[int, int, int]:
        """
        Divide the ratio by n_groups.
        Args:
            n_groups: The number of groups to divide into.

        Returns:
            A tuple of (n_train, n_val, n_test) representing the number of groups for training, validation, and testing.
        """
        # 浮点比例允许微小误差，避免 0.1+0.2 这类精度问题导致误判。
        if abs(self._sum() - 1.0) > 1e-8:
            raise ValueError(
                f"Invalid split ratio: train+val+test must equal 1.0, got {self._sum():.8f}"
            )

        # 至少保证有训练样本
        if n_groups <= 1:
            return 1, 0, 0
        if n_groups == 2:
            return 1, 1, 0
        if n_groups == 3:
            return 1, 1, 1
        n_val = max(1, int(round(n_groups * self.val)))
        n_test = max(1, int(round(n_groups * self.test)))
        n_train = n_groups - n_val - n_test
        # 如果训练集数量不足，优先从验证集减少，再从测试集减少
        if n_train < 1:
            need = 1 - n_train
            if n_val >= n_test and n_val - need >= 1:
                n_val -= need
            else:
                n_test -= need
            n_train = 1

        return n_train, n_val, n_test


def gen_index(ratio: DivideRatio = DivideRatio(0.8, 0.1, 0.1),
              order_name: str = "",
              seed: int = 42,
              split_id: str = "",
              activate: bool = False,
              data_root: str = "./artifacts/data",
              ) -> None:
    """
    从 label.csv 生成 split.csv, 包含每个样本所属的 split(train/val/test)切分时按物种(label)分层，保证每个 split 内的类别分布相似；
    同一 group_id 的样本会被分配到同一 split，避免数据泄漏。
    Args:
        order_name: 指定‘目’，如果不指定，则使用全部数据
        ratio: 训练集，验证集，测试集的比例，默认 8:1:1
        seed: 随机数种子
        split_id: 切分版本标识，为空时使用 active_split 或 default
        activate: 是否将本次 split_id 写入 active_split 标记文件
    """
    # 固定随机种子，保证每次切分可复现
    rng = random.Random(seed)
    label_path = Path(data_root) / "label.csv"
    # 读取标签文件，后续所有切分都基于该文件展开
    if not label_path.exists():
        logger.error(f"No label file found at {label_path}, please generate by exec: uv run -m scripts.gen_label")

    df: pl.DataFrame = pl.read_csv(label_path)
    # 可选按“目”过滤；为空时使用全量数据
    if order_name != "":
        df = df.filter(pl.col("order") == order_name)
        if df.height == 0:
            raise ValueError(f"Order name {order_name} not found in label.csv")

    # 去重到 group 级别：同一 group_id 只会被分配到一个 split
    group_df: pl.DataFrame = df.select(["label", "group_id"]).unique()

    assignments: list[dict[str, str]] = []
    # 按物种(label)分层切分，避免类别分布严重失衡
    for per_label in group_df.partition_by("label", as_dict=False):
        label = per_label["label"][0]
        group_ids: list[str] = per_label["group_id"].to_list()
        # 每个物种使用自己的 group 数量计算分配
        n_groups = len(group_ids)

        # 先打乱 group 顺序，再按比例切出 train/val/test
        rng.shuffle(group_ids)
        n_train, n_val, n_test = ratio.divide(n_groups)
        train_ids = group_ids[:n_train]
        val_ids = group_ids[n_train:n_train + n_val]
        test_ids = group_ids[n_train + n_val:n_train + n_val + n_test]

        # 记录 group_id 到 split 的映射，后面回填到图片级数据
        assignments.extend({"group_id": gid, "split": "train"} for gid in train_ids)
        assignments.extend({"group_id": gid, "split": "val"} for gid in val_ids)
        assignments.extend({"group_id": gid, "split": "test"} for gid in test_ids)

        logger.info(
            f"[{label}] groups={len(group_ids)} train={n_train} val={n_val} test={n_test}"
        )
    # 形成统一的 group->split 映射表
    split_map = pl.DataFrame(assignments)

    # 回填 split 到原始 df：此时保留了 label.csv 的全部属性
    full_df = df.join(split_map, on="group_id", how="left")

    # 防御性检查：不允许存在未分配 split 的样本
    if full_df["split"].null_count() > 0:
        raise RuntimeError("Found rows without split assignment.")

    marker_path = Path(data_root) / "active_split.txt"
    marker_value = marker_path.read_text(encoding="utf-8").strip() if marker_path.exists() else ""
    resolved_split_id = split_id.strip() if split_id else marker_value
    if resolved_split_id == "":
        resolved_split_id = "default"

    split_path = Path(data_root) / "splits" / resolved_split_id / "split.csv"
    split_path.parent.mkdir(parents=True, exist_ok=True)

    # split.csv 作为总清单；train/val/test 为下游训练直接使用文件
    full_df.write_csv(split_path)

    meta_path = split_path.parent / "meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "split_id": resolved_split_id,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "seed": seed,
                "order": order_name,
                "ratio": {"train": ratio.train, "val": ratio.val, "test": ratio.test},
                "rows": full_df.height,
                "label_path": str(label_path),
                "split_path": str(split_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if activate:
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(resolved_split_id, encoding="utf-8")
        logger.info(f"active split updated: {resolved_split_id} -> {marker_path}")

    logger.info(f"split saved: {split_path}")
    logger.info(f"meta saved: {meta_path}")

    return None


@app.command()
def main(
    data_root: str = typer.Option("./artifacts/data", help="产物数据根目录"),
    log_dir: str = typer.Option("./artifacts/logs", help="日志输出目录"),
    order: str = typer.Option("", help="指定目名称，如 直翅目；为空则使用全量数据"),
    train: float = typer.Option(0.8, help="训练集比例"),
    val: float = typer.Option(0.1, help="验证集比例"),
    test: float = typer.Option(0.1, help="测试集比例"),
    seed: int = typer.Option(42, help="随机种子"),
    split_id: str = typer.Option("", help="切分版本 ID，留空则读取 active_split.txt"),
    activate: bool = typer.Option(False, "--activate/--no-activate", help="是否将本次切分写为当前激活版本"),
) -> None:
    init_logger(log_dir)
    gen_index(
        ratio=DivideRatio(train=train, val=val, test=test),
        order_name=order,
        seed=seed,
        split_id=split_id,
        activate=activate,
        data_root=data_root,
    )


if __name__ == "__main__":
    app()

