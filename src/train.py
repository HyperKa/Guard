import polars as pl
from catboost import CatBoostClassifier, Pool
from src.config import DROP_COLS


def train_model(train_df: pl.DataFrame):
    split_idx = int(len(train_df) * 0.8)

    features = [c for c in train_df.columns if c not in DROP_COLS]
    cat_features = [c for c in features if train_df[c].dtype == pl.Categorical]

    print("Подготовка Pool...")

    train_pool = Pool(
        data=train_df[:split_idx].select(features).to_pandas(),
        label=train_df[:split_idx]["target"].to_numpy(),
        cat_features=cat_features
    )

    val_pool = Pool(
        data=train_df[split_idx:].select(features).to_pandas(),
        label=train_df[split_idx:]["target"].to_numpy(),
        cat_features=cat_features
    )

    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        task_type="GPU",
        devices='0',
        eval_metric='PRAUC',
        metric_period=50,

        auto_class_weights='Balanced',

        border_count=32,
        max_ctr_complexity=1,
        gpu_ram_part=0.9,
        verbose=100
    )

    model.fit(train_pool, eval_set=val_pool)
    return model, features