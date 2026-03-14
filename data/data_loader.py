# from pathlib import Path

# import polars as pl
from pydantic import BaseModel


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


class DataSet:
    def __init__(self, dataset_path: str):
        self.dataset_path: str = dataset_path

    def collect(self):
        pass
