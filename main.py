from src.data_loader import get_train_data
from src.features import build_features
from src.train import train_model
from src.config import TEST, PRETEST
import polars as pl


def main():
    print("Загрузка и расчет признаков для Train...")
    train_lazy = get_train_data(limit_parts=3)
    full_train = build_features(train_lazy).collect()

    print("Расчет рисков MCC (Target Encoding)...")
    mcc_risk = full_train.group_by("mcc_code").agg(pl.col("target").mean())
    mcc_map = dict(zip(mcc_risk["mcc_code"], mcc_risk["target"]))

    full_train = full_train.with_columns(
        pl.col("mcc_code").replace(mcc_map, default=0).cast(pl.Float32).alias("mcc_code_risk")
    )

    # вот тут можно побаловаться с объемом, но сейчас на 0.2 и 0.5 V разницы нет из-за дизбаланса
    print("Сэмплирование выборки...")
    frauds = full_train.filter(pl.col("target") > 0)
    normals = full_train.filter(pl.col("target") == 0).sample(fraction=0.5, seed=42)
    train_df = pl.concat([frauds, normals]).sort("event_dttm")

    print(f"Итоговый размер Train: {len(train_df)} строк (Фрод: {len(frauds)})")

    print("Запуск обучения...")
    model, feature_names = train_model(train_df)

    print("Подготовка теста...")
    test_context_lazy = pl.concat([pl.scan_parquet(PRETEST), pl.scan_parquet(TEST)])

    test_df = build_features(test_context_lazy, target_mapping={"mcc_code": mcc_map}).collect()

    real_test_ids = pl.read_parquet(TEST).select("event_id")
    test_final = test_df.join(real_test_ids, on="event_id", how="inner")

    print("Создание сабмита...")
    preds = model.predict_proba(test_final.select(feature_names).to_pandas())[:, 1]

    submission = pl.DataFrame({
        "event_id": test_final["event_id"],
        "predict": preds
    })

    submission.write_csv("submission.csv")
    print("Готово! submission.csv создан.")


if __name__ == "__main__":
    main()