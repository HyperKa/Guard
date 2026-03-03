import polars as pl

df = pl.read_parquet('data/train_labels.parquet')

# первые 5 строк
print(df.head())

# сама разметка
print(df.schema)