from pathlib import Path

import polars as pl

from utils import Config

df: pl.DataFrame = pl.read_csv(
    Path(Config.parse().data_root) / "label.csv",
    schema_overrides={
        "group": pl.String,
    },
)

print(f"sample count: {df.height}")


print("species count:", df.select("species").unique().height)
