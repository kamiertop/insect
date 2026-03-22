import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import polars as pl

from data.data_loader import DataItem
from utils import Config


def gen_label() -> None:
    """
    处理数据集，生成标签文件 "label.csv"
    """
    config = Config.parse()

    items: list[DataItem] = []

    # 纲
    fs_class = Path(config.datasets_path)

    for order in fs_class.iterdir():  # 目
        for family in order.iterdir():  # 科
            for genus in family.iterdir():  # 属
                for species in genus.iterdir():  # 种
                    for sex_or_specimen_id in species.iterdir():  # 性别或者直接是标本号
                        for img_or_group in sex_or_specimen_id.iterdir():
                            if img_or_group.is_file():
                                item = DataItem(
                                    order=order.name,
                                    family=family.name,
                                    genus=genus.name,
                                    species=species.name,
                                    label=species.name,
                                    img_name=img_or_group.name,
                                    path=str(img_or_group),
                                    sex="unknown",
                                    group=sex_or_specimen_id.name,
                                    group_id=f"{species.name}_{sex_or_specimen_id.name}",
                                )
                                items.append(item)
                            else:
                                group: Path = img_or_group
                                for img in group.iterdir():
                                    item = DataItem(
                                        order=order.name,
                                        family=family.name,
                                        genus=genus.name,
                                        species=species.name,
                                        label=species.name,
                                        img_name=img.name,
                                        path=str(img),
                                        sex=sex_or_specimen_id.name,
                                        group=group.name,
                                        group_id=f"{species.name}_{group.name}",
                                    )
                                    items.append(item)

    df: pl.DataFrame = pl.DataFrame(items)

    out_path = Path(config.data_root) / "label.csv"
    df.write_csv(out_path)

    print(f"output path: {out_path}")

    return None


if __name__ == "__main__":
    print("处理数据集，生成标签文件...")
    gen_label()
    print("处理完成，标签生成完成")
