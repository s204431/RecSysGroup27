from pathlib import Path
import polars as pl

from ebrec.utils._descriptive_analysis import (
    min_max_impression_time_behaviors,
    min_max_impression_time_history,
)
from ebrec.utils._polars import slice_join_dataframes
from ebrec.utils._behaviors import (
    create_binary_labels_column,
    sampling_strategy_wu2019,
    truncate_history,
)
from ebrec.utils._constants import *
from ebrec.utils._python import compute_npratio

PATH = Path(__file__).parent.resolve().joinpath("ebnerd_data")
TRAIN_VAL_SPLIT = f"ebnerd_demo"  # [ebnerd_demo, ebnerd_small, ebnerd_large]
TEST_SPLIT = f"ebnerd_testset"

df_behaviors_train = df_behaviors = pl.scan_parquet(
    PATH.joinpath(TRAIN_VAL_SPLIT, "train", "behaviors.parquet")
)
df_history_train = df_behaviors = pl.scan_parquet(
    PATH.joinpath(TRAIN_VAL_SPLIT, "train", "history.parquet")
)
df_behaviors_val = df_behaviors = pl.scan_parquet(
    PATH.joinpath(TRAIN_VAL_SPLIT, "validation", "behaviors.parquet")
)
df_history_val = df_behaviors = pl.scan_parquet(
    PATH.joinpath(TRAIN_VAL_SPLIT, "validation", "history.parquet")
)
df_behaviors_test = df_behaviors = pl.scan_parquet(
    PATH.joinpath(TEST_SPLIT, "test", "behaviors.parquet")
)
df_history_test = df_behaviors = pl.scan_parquet(
    PATH.joinpath(TEST_SPLIT, "test", "history.parquet")
)
df_articles = pl.scan_parquet(PATH.joinpath(TEST_SPLIT, "articles.parquet"))
print(f"History: {min_max_impression_time_history(df_history_train).collect()}")
print(f"Behaviors: {min_max_impression_time_behaviors(df_behaviors_train).collect()}")



df_history = df_history_train.select(
    DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL
).pipe(
    truncate_history,
    column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    history_size=30,
    padding_value=0,
    enable_warning=False,
)
print(df_history.head(5).collect())


df = slice_join_dataframes(
    df1=df_behaviors_train.collect(),
    df2=df_history_train.collect(),
    on=DEFAULT_USER_COL,
    how="left",
)
print(df.head(5))



print(df.select(DEFAULT_CLICKED_ARTICLES_COL, DEFAULT_INVIEW_ARTICLES_COL).pipe(
    create_binary_labels_column, shuffle=True, seed=123
).with_columns(pl.col("labels").list.len().name.suffix("_len")).head(5))




NPRATIO = 2
print(df.select(DEFAULT_CLICKED_ARTICLES_COL, DEFAULT_INVIEW_ARTICLES_COL).pipe(
    sampling_strategy_wu2019,
    npratio=NPRATIO,
    shuffle=False,
    with_replacement=True,
    seed=123,
).pipe(create_binary_labels_column, shuffle=True, seed=123).with_columns(
    pl.col("labels").list.len().name.suffix("_len")
).head(
    5
))