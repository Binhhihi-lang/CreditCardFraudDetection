"""Suy luận gian lận cho giao dịch mới bằng mô hình đã huấn luyện.

Cách dùng từ dòng lệnh::

    # Dự đoán cho file CSV (đủ các cột Time, V1..V28, Amount; cột Class nếu có sẽ bị bỏ qua)
    python predict.py --input data/sample_transactions.csv --output predictions.csv

    # Dự đoán cho file JSON (danh sách các giao dịch hoặc một giao dịch)
    python predict.py --input giao_dich.json

    # Đổi ngưỡng quyết định (mặc định dùng ngưỡng đã lưu cùng mô hình)
    python predict.py --input data/sample_transactions.csv --threshold 0.3

Cách dùng từ Python::

    from predict import predict
    result = predict(df)          # df có các cột Time, V1..V28, Amount

Kết quả có hai cột thêm vào: ``fraud_probability`` (xác suất gian lận) và
``is_fraud`` (1 nếu xác suất >= ngưỡng).
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

# Import này cần thiết để joblib nạp được lớp FraudFeatureEngineer đã lưu trong pipeline.
import fraud_pipeline  # noqa: F401
from fraud_pipeline import RAW_FEATURES

MODEL_PATH = Path(__file__).resolve().parent / "models" / "best_model.joblib"


@lru_cache(maxsize=1)
def load_model(model_path: str | Path = MODEL_PATH) -> dict:
    """Nạp gói mô hình (pipeline, ngưỡng, siêu dữ liệu). Chỉ nạp một lần."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {model_path}. Hãy chạy notebook 02_Modeling_Evaluation.ipynb "
            "để huấn luyện và xuất mô hình."
        )
    return joblib.load(model_path)


def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    """Kiểm tra dữ liệu đầu vào và trả về đúng các cột cần cho mô hình."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Đầu vào phải là pandas DataFrame.")
    if df.empty:
        raise ValueError("Dữ liệu đầu vào rỗng.")
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu các cột bắt buộc: {missing}")
    features = df[RAW_FEATURES].apply(pd.to_numeric, errors="coerce")
    n_bad = int(features.isna().sum().sum())
    if n_bad:
        # Không dừng chương trình: imputer trong pipeline sẽ điền giá trị thiếu.
        print(f"Cảnh báo: {n_bad} ô thiếu/không phải số, sẽ được điền bằng imputer của pipeline.",
              file=sys.stderr)
    return features


def predict(df: pd.DataFrame, threshold: float | None = None,
            model_path: str | Path = MODEL_PATH) -> pd.DataFrame:
    """Dự đoán gian lận cho các giao dịch trong ``df``.

    Parameters
    ----------
    df : DataFrame
        Các cột bắt buộc: ``Time``, ``V1``..``V28``, ``Amount``.
    threshold : float, tùy chọn
        Ngưỡng quyết định. Mặc định dùng ngưỡng tối ưu đã lưu cùng mô hình.

    Returns
    -------
    DataFrame
        Bản sao của ``df`` kèm cột ``fraud_probability`` và ``is_fraud``.
    """
    bundle = load_model(str(model_path))
    thr = bundle["threshold"] if threshold is None else float(threshold)
    if not 0.0 <= thr <= 1.0:
        raise ValueError("Ngưỡng phải nằm trong [0, 1].")
    features = validate_input(df)
    proba = bundle["pipeline"].predict_proba(features)[:, 1]
    result = df.copy()
    result["fraud_probability"] = proba
    result["is_fraud"] = (proba >= thr).astype(int)
    return result


def read_transactions(path: str | Path) -> pd.DataFrame:
    """Đọc giao dịch từ file CSV hoặc JSON."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file đầu vào: {path}")
    if path.suffix.lower() == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame([data] if isinstance(data, dict) else data)
    return pd.read_csv(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phát hiện giao dịch thẻ tín dụng gian lận.")
    parser.add_argument("--input", required=True, help="File CSV hoặc JSON chứa giao dịch")
    parser.add_argument("--output", help="File CSV để lưu kết quả (tùy chọn)")
    parser.add_argument("--threshold", type=float, help="Ngưỡng quyết định (mặc định: ngưỡng đã lưu)")
    args = parser.parse_args(argv)

    try:
        bundle = load_model(str(MODEL_PATH))
        thr = bundle["threshold"] if args.threshold is None else args.threshold
        result = predict(read_transactions(args.input), threshold=args.threshold)
    except (FileNotFoundError, ValueError, TypeError, pd.errors.ParserError) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1

    n_fraud = int(result["is_fraud"].sum())
    print(f"Mô hình: {bundle['model_name']} | ngưỡng = {thr:.4f}")
    print(f"Đã phân tích {len(result)} giao dịch, phát hiện {n_fraud} giao dịch nghi gian lận.")
    show_cols = [c for c in ("Time", "Amount", "fraud_probability", "is_fraud", "Class") if c in result.columns]
    print(result[show_cols].sort_values("fraud_probability", ascending=False).head(10)
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if "Class" in result.columns:
        tp = int(((result["is_fraud"] == 1) & (result["Class"] == 1)).sum())
        print(f"(Có nhãn Class trong file: bắt được {tp}/{int(result['Class'].sum())} giao dịch gian lận thật.)")
    if args.output:
        result.to_csv(args.output, index=False)
        print(f"Đã lưu kết quả vào {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
