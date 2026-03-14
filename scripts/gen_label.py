import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import polars as pl

from data.data_loader import DataItem
from utils import Config, init_logger


def gen_label() -> None:
    init_logger()
    config = Config.parse()

    items: list[DataItem] = []

    # 纲
    fs_class = Path(config.datasets_path)

    for order in fs_class.iterdir():  # 目
        for family in order.iterdir():  # 科
            for genus in family.iterdir():  # 属
                for species in genus.iterdir():  # 种
                    for sex_or_specimen_id in species.iterdir():
                        for img_or_sex in sex_or_specimen_id.iterdir():
                            if img_or_sex.is_file():
                                item = DataItem(
                                    order=order.name,
                                    family=family.name,
                                    genus=genus.name,
                                    species=species.name,
                                    label=species.name,
                                    img_name=img_or_sex.name,
                                    path=str(img_or_sex),
                                    sex="unknown",
                                    group=sex_or_specimen_id.name,
                                    group_id=f"{species.name}_unknown_{sex_or_specimen_id.name}",
                                )
                                items.append(item)
                            else:
                                sex = img_or_sex
                                for img in sex.iterdir():
                                    item = DataItem(
                                        order=order.name,
                                        family=family.name,
                                        genus=genus.name,
                                        species=species.name,
                                        label=species.name,
                                        img_name=img.name,
                                        path=str(img),
                                        sex=sex.name,
                                        group=sex_or_specimen_id.name,
                                        group_id=f"{species.name}_{sex.name}_{sex_or_specimen_id.name}",
                                    )
                                    items.append(item)

    df: pl.DataFrame = pl.DataFrame(items)

    out_path = Path(config.data_root) / "label.csv"
    df.write_csv(out_path)

    return None


if __name__ == "__main__":
    gen_label()
