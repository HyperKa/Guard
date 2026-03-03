from pathlib import Path

# Пути
DATA_DIR = Path("data")
PROCESSED_DIR = Path("processed")

# Файлы
TRAIN_PARTS = [DATA_DIR / f"train_part_{i}.parquet" for i in [1, 2, 3]]
TRAIN_LABELS = DATA_DIR / "train_labels.parquet"
PRETEST = DATA_DIR / "pretest.parquet"
TEST = DATA_DIR / "test.parquet"

# Признаки
CAT_COLS = ["mcc_code", "currency_iso_cd", "event_type_nm", "channel_indicator_type", "operating_system_type"]
DROP_COLS = ["event_id", "customer_id", "event_dttm", "target", "session_id"]