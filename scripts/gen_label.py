import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import polars as pl
import typer

from dataset.data_loader import DataItem

app = typer.Typer(add_completion=False, help="扫描数据集并生成 label.csv")


def gen_label(datasets_path: str, data_root: str) -> None:
    """
    处理数据集，生成标签文件 "label.csv"
    """
    items: list[DataItem] = []

    # 按 目/科/属/种/性别(or标本) 的目录层级收集图片并构建标签表。
    fs_class = Path(datasets_path)

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

    out_path = Path(data_root) / "label.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(out_path)

    print(f"output path: {out_path}")

    return None


@app.command()
def main(
    datasets_path: str = typer.Option("./data/datasets/昆虫纲", help="原始数据集目录"),
    data_root: str = typer.Option("./artifacts/data", help="产物数据根目录"),
) -> None:
    print("处理数据集，生成标签文件...")
    gen_label(datasets_path=datasets_path, data_root=data_root)
    print("处理完成，标签生成完成")


if __name__ == "__main__":
    app()

