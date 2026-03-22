import argparse
import random
from pathlib import Path

import polars as pl
from loguru import logger

from utils import Config, init_logger


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
        assert self._sum() == 1.0

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
              seed: int = 42
              ) -> None:
    """
    从 label.csv 生成 train.csv, val.csv, test.csv
    Args:
        order_name: 指定‘目’，如果不指定，则使用全部数据
        ratio: 训练集，验证集，测试集的比例，默认 8:1:1
        seed: 随机数种子
    """
    # 固定随机种子，保证每次切分可复现
    rng = random.Random(seed)
    cfg = Config.parse()
    label_path = Path(cfg.data_root) / "label.csv"
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

    # 基于 split 字段切出三个集合，且保留所有原始列
    train_df = full_df.filter(pl.col("split") == "train")
    val_df = full_df.filter(pl.col("split") == "val")
    test_df = full_df.filter(pl.col("split") == "test")

    data_root = Path(cfg.data_root)
    split_path = data_root / "split.csv"
    train_path = data_root / "train.csv"
    val_path = data_root / "val.csv"
    test_path = data_root / "test.csv"

    # split.csv 作为总清单；train/val/test 为下游训练直接使用文件
    full_df.write_csv(split_path)
    train_df.write_csv(train_path)
    val_df.write_csv(val_path)
    test_df.write_csv(test_path)

    logger.info(f"split saved: {split_path}")
    logger.info(f"train saved: {train_path}, rows={train_df.height}")
    logger.info(f"val saved:   {val_path}, rows={val_df.height}")
    logger.info(f"test saved:  {test_path}, rows={test_df.height}")

    return None


if __name__ == "__main__":
    init_logger()
    
    # 从命令行读取 order_name，不指定时默认使用全部数据
    parser = argparse.ArgumentParser(description="从 label.csv 生成 train/val/test 索引文件")
    parser.add_argument(
        "--order", "-o",
        type=str,
        default="",
        help="指定'目'名称，如'直翅目'；不指定时使用全部数据"
    )
    parser.add_argument(
        "--train",
        type=float,
        default=0.8,
        help="训练集比例，默认 0.8"
    )
    parser.add_argument(
        "--val",
        type=float,
        default=0.1,
        help="验证集比例，默认 0.1"
    )
    parser.add_argument(
        "--test",
        type=float,
        default=0.1,
        help="测试集比例，默认 0.1"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，默认 42，用于可复现性"
    )
    
    args = parser.parse_args()
    
    ratio = DivideRatio(train=args.train, val=args.val, test=args.test)
    gen_index(ratio=ratio, order_name=args.order, seed=args.seed)
