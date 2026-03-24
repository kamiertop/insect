from pathlib import Path
from typing import Callable

import polars as pl
import torch.cuda
from PIL import Image
from pydantic import BaseModel
from torch.utils.data import Dataset, DataLoader

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

    def __init__(self, config: Config, split: str = 'train', transform: Callable = None):

        assert split in ['train', 'val', 'test']
        self.transform = transform
        self.dataset_path: str = config.datasets_path
        self.df: pl.DataFrame = pl.read_csv(Path(config.data_root) / f"{split}.csv")
        self.data: list = []

        if self.df.height == 0:
            raise ValueError(f"{split} has no rows")

        classes: list[str] = self.df.select("label").drop_nulls().unique().to_series().to_list()
        self.class_to_idx: dict[str, int] = {}
        self.idx_to_class: dict[int, str] = {}
        for i, name in enumerate(classes):
            self.class_to_idx[name] = i
            self.idx_to_class[i] = name

        self.samples: list[dict] = self.df.select(["path", "label", "group_id"]).to_dicts()

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)

    def __getitem__(self, index: int):
        row: dict = self.samples[index]
        img = Image.open(row["path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        class_idx: int = self.class_to_idx[row["label"]]

        return img, class_idx, {"path": row["path"], "group_id": row["group_id"]}


def build_dataloader(dataset: InsectDataset, shuffle=False, batch_size=32, num_workers=4) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_initialized()
    )
