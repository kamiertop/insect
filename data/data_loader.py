from pathlib import Path

import polars as pl
from pydantic import BaseModel
from torch.utils.data import Dataset

from utils import Config


class DataItem(BaseModel):
    order: str  # 目
    family: str  # 科
    genus: str  # 属
    species: str  # 种
    label: str  # 物种作为标签
    sex: str
    group: str  # 样本编号
    img_name: str  # 文件名
    group_id: str  # 文件名_id
    path: str  # 路径


class InsectDataset(Dataset):
    """
    Attributes:
        class_num: 有多少类别，支持通过order_name指定‘目’
    """

    def __init__(self, config: Config, order_name: str = ""):
        self.dataset_path: str = config.datasets_path
        self.df: pl.DataFrame = pl.read_csv(Path(config.data_root) / "label.csv")
        self.data: list = []
        self.class_num: int = self.class_num_by_order(order_name=order_name)

    def class_num_by_order(self, order_name: str) -> int:
        """
        获取一个‘目’下面有多少种 (即 label 的唯一值数量)
        """

        return (
            self
            .df
            .select(["order", "label"])
            .filter(pl.col("order") == order_name)
            .select("label")
            .n_unique()
        )

    def __len__(self) -> int:
        return len(self.data)

    # TODO
    def __getitem__(self, index: int):
        pass
