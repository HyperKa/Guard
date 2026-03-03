import polars as pl


def build_features(df: pl.LazyFrame, target_mapping: dict = None) -> pl.LazyFrame:
    if df.schema["event_dttm"] == pl.String:
        df = df.with_columns(pl.col("event_dttm").str.to_datetime())

    df = df.sort(["customer_id", "event_dttm"])

    # первая версия признаков
    existing_cols = df.schema.keys()
    if "hour" not in existing_cols:
        df = df.with_columns([
            pl.col("event_dttm").dt.hour().alias("hour"),
            pl.col("event_dttm").dt.weekday().alias("day_of_week"),
            pl.col("operaton_amt").mean().over("customer_id").alias("user_avg_amt"),
            pl.col("event_id").count().over("customer_id").alias("user_trans_count"),
            (pl.col("operaton_amt") / (pl.col("operaton_amt").mean().over("customer_id") + 1)).alias(
                "amt_to_avg_ratio"),
            (pl.col("event_dttm") - pl.col("event_dttm").shift(1).over("customer_id")).dt.total_seconds().alias(
                "seconds_since_last_op"),
            pl.col("compromised").cast(pl.Int8),
            pl.col("web_rdp_connection").cast(pl.Int8),
            pl.col("operaton_amt").std().over("customer_id").alias("user_std_amt"),
            pl.col("mcc_code").n_unique().over("customer_id").alias("user_mcc_count"),
            pl.col("event_id").count().over("session_id").alias("ops_per_session")
        ])

    if target_mapping:
        for col, mapping in target_mapping.items():
            df = df.with_columns(
                pl.col(col).replace(mapping, default=0).cast(pl.Float32).alias(f"{col}_risk")
            )

    return df.with_columns([
        pl.col(pl.String).fill_null("NONE").cast(pl.Categorical),
        pl.col(pl.NUMERIC_DTYPES).exclude(["customer_id", "event_id"]).cast(pl.Float32).fill_null(0)
    ])