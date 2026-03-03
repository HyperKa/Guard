import polars as pl
from src.config import TRAIN_PARTS, TRAIN_LABELS


def get_train_data(limit_parts=None):
    parts = TRAIN_PARTS[:limit_parts] if limit_parts else TRAIN_PARTS

    # На всякий случай сортировка вставлена, как поведут себя отсортированные данные при конкатенации из 3 частей - неясно до конца
    ops = pl.concat([pl.scan_parquet(p) for p in parts]).sort("event_dttm")
    labels = pl.scan_parquet(TRAIN_LABELS)

    df = ops.join(labels.select(["event_id", "target"]), on="event_id", how="left")
    return df.with_columns(pl.col("target").fill_null(0))


