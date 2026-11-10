"""Thành phần dùng chung cho dự án Credit Card Fraud Detection.

Module này gom toàn bộ phần xử lý dữ liệu và xây dựng Pipeline để các notebook
(01_EDA_Preprocessing, 02_Modeling_Evaluation) và ``predict.py`` dùng cùng một
đoạn mã. Nhờ vậy lúc huấn luyện và lúc suy luận cho dữ liệu mới, dữ liệu luôn đi
qua đúng một luồng xử lý.

Nội dung chính:
    * ``load_data`` / ``clean_data`` / ``split_data``: đọc, làm sạch và chia dữ liệu.
    * ``FraudFeatureEngineer``: transformer kỹ nghệ đặc trưng (không có tham số học
      từ dữ liệu nên không gây rò rỉ).
    * ``build_preprocessor`` / ``build_pipeline``: ColumnTransformer và Pipeline
      hoàn chỉnh (kỹ nghệ đặc trưng -> impute/scale/encode -> lấy mẫu -> mô hình).

Quy ước chống rò rỉ dữ liệu (data leakage):
    * Làm sạch (bỏ dòng trùng) và chia train/test được thực hiện TRƯỚC mọi thao tác
      học tham số từ dữ liệu.
    * Imputer, scaler, encoder và sampler (SMOTE, undersampling) đều nằm trong
      Pipeline nên chỉ được fit trên phần train của từng fold.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

# --------------------------------------------------------------------------- #
# Hằng số
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "Class"
V_COLUMNS = [f"V{i}" for i in range(1, 29)]
RAW_FEATURES = ["Time"] + V_COLUMNS + ["Amount"]

# Đường dẫn mặc định tới dữ liệu gốc: <thư mục này>/data/creditcard.csv
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_DIR / "data" / "creditcard.csv"

SECONDS_PER_DAY = 24 * 3600
# Các biến V có khả năng phân biệt gian lận mạnh nhất, xác định từ EDA trên
# tập TRAIN (xem notebook 01, mục 6). Dùng để tạo đặc trưng tương tác.
TOP_V = ("V14", "V12", "V10", "V17")
PERIOD_LABELS = ["night", "morning", "afternoon", "evening"]


# --------------------------------------------------------------------------- #
# Đọc, làm sạch, chia dữ liệu
# --------------------------------------------------------------------------- #
def load_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Đọc file ``creditcard.csv`` và kiểm tra các cột bắt buộc."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu tại {path}. Xem data/README.md để tải "
            "creditcard.csv từ Kaggle và đặt vào thư mục data/."
        )
    df = pd.read_csv(path)
    missing_cols = [c for c in RAW_FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Thiếu các cột bắt buộc: {missing_cols}")
    return df


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Làm sạch dữ liệu thô: bỏ dòng trùng lặp hoàn toàn.

    Dòng trùng được loại trước khi chia train/test để cùng một giao dịch không
    xuất hiện ở cả hai tập. Trả về (dữ liệu đã làm sạch, báo cáo làm sạch).
    """
    n_before = len(df)
    n_missing = int(df.isna().sum().sum())
    n_dup = int(df.duplicated().sum())
    n_dup_fraud = int(df.loc[df[TARGET] == 1].duplicated().sum())
    clean = df.drop_duplicates().reset_index(drop=True)
    report = {
        "rows_before": n_before,
        "rows_after": len(clean),
        "missing_values": n_missing,
        "duplicates_removed": n_dup,
        "duplicates_in_fraud": n_dup_fraud,
        "fraud_after": int(clean[TARGET].sum()),
        "fraud_rate_after": float(clean[TARGET].mean()),
    }
    return clean, report


def split_data(
    df: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE
):
    """Chia train/test phân tầng theo nhãn, giữ nguyên tỷ lệ gian lận."""
    X = df[RAW_FEATURES]
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def load_clean_split(path: str | Path = DEFAULT_DATA_PATH):
    """Đọc -> làm sạch -> chia train/test. Trả về X_train, X_test, y_train, y_test.

    Hai notebook gọi cùng hàm này với random_state cố định nên nhận được đúng
    cùng một phép chia.
    """
    df, _ = clean_data(load_data(path))
    return split_data(df)


# --------------------------------------------------------------------------- #
# Kỹ nghệ đặc trưng
# --------------------------------------------------------------------------- #
class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """Tạo đặc trưng mới từ 30 cột gốc (Time, V1-V28, Amount).

    Transformer này *không học tham số từ dữ liệu* (``fit`` không làm gì), mọi
    đặc trưng đều là hàm xác định của chính dòng đó, nên không thể gây rò rỉ
    dữ liệu dù dùng trong cross-validation.

    Đặc trưng đầu ra:
        * ``V1..V28``: giữ nguyên.
        * ``Amount``: số tiền gốc.
        * ``Amount_log``: ``log1p(Amount)`` để giảm lệch phải.
        * ``Amount_small``: cờ giao dịch số tiền nhỏ (< 1), dạng "dò thẻ".
        * ``Hour``, ``Hour_sin``, ``Hour_cos``: giờ trong ngày (tương đối so với giao dịch
          đầu tiên của bộ dữ liệu) và mã hóa vòng tròn.
        * ``Period``: khung giờ (night/morning/afternoon/evening), biến phân loại.
        * ``V_norm``: chuẩn L2 của V1..V28, đo mức bất thường tổng thể.
        * ``V14_x_V12``, ``V14_x_V10``, ``V14_x_V17``, ``V12_x_V10``: tương tác
          giữa các biến V phân biệt mạnh nhất.

    Cột ``Time`` (số giây kể từ giao dịch đầu tiên của bộ dữ liệu) bị loại vì giá
    trị tuyệt đối của nó không tổng quát hóa được sang dữ liệu mới.
    """

    def fit(self, X, y=None):
        self._check_columns(X)
        self.feature_names_in_ = list(X.columns)
        return self

    @staticmethod
    def _check_columns(X) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FraudFeatureEngineer cần đầu vào là pandas DataFrame.")
        missing = [c for c in RAW_FEATURES if c not in X.columns]
        if missing:
            raise ValueError(f"Dữ liệu đầu vào thiếu các cột: {missing}")

    def transform(self, X):
        self._check_columns(X)
        out = X[V_COLUMNS + ["Amount"]].copy()

        out["Amount_log"] = np.log1p(X["Amount"].clip(lower=0))
        out["Amount_small"] = (X["Amount"] < 1).astype(float)

        hour = (X["Time"] % SECONDS_PER_DAY) / 3600.0
        out["Hour"] = hour
        out["Hour_sin"] = np.sin(2 * np.pi * hour / 24)
        out["Hour_cos"] = np.cos(2 * np.pi * hour / 24)
        out["Period"] = pd.cut(
            hour,
            bins=[-0.001, 6, 12, 18, 24.001],
            labels=PERIOD_LABELS,
        ).astype(str)

        out["V_norm"] = np.sqrt((X[V_COLUMNS] ** 2).sum(axis=1))
        out["V14_x_V12"] = X["V14"] * X["V12"]
        out["V14_x_V10"] = X["V14"] * X["V10"]
        out["V14_x_V17"] = X["V14"] * X["V17"]
        out["V12_x_V10"] = X["V12"] * X["V10"]
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(self.transform(self._dummy_row()).columns)

    @staticmethod
    def _dummy_row() -> pd.DataFrame:
        """Một dòng giả để suy ra tên cột đầu ra."""
        return pd.DataFrame([dict.fromkeys(RAW_FEATURES, 1.0)])


# --------------------------------------------------------------------------- #
# ColumnTransformer & Pipeline
# --------------------------------------------------------------------------- #
# Nhóm cột sau bước kỹ nghệ đặc trưng
V_AND_INTERACTION = V_COLUMNS + ["V_norm", "V14_x_V12", "V14_x_V10",
                                 "V14_x_V17", "V12_x_V10"]
AMOUNT_TIME_NUM = ["Amount", "Amount_log", "Hour", "Hour_sin", "Hour_cos"]
BINARY_COLS = ["Amount_small"]
CATEGORICAL_COLS = ["Period"]


def build_preprocessor(scale_v: str = "standard") -> ColumnTransformer:
    """ColumnTransformer: impute + chuẩn hóa + mã hóa cho từng nhóm cột.

    * Nhóm V (đã là thành phần PCA) và các tương tác: impute median + StandardScaler.
    * Nhóm Amount/Hour: impute median + RobustScaler (chịu được ngoại lai, vì số
      tiền có đuôi rất dài và giao dịch gian lận nằm ở các giá trị cực trị).
    * Cờ nhị phân: impute most_frequent, giữ nguyên.
    * Biến phân loại ``Period``: impute most_frequent + OneHotEncoder
      (``handle_unknown='ignore'`` để dữ liệu mới không làm vỡ pipeline).

    Dữ liệu gốc không thiếu giá trị nào, nhưng imputer vẫn được đặt vào để
    pipeline an toàn khi gặp dữ liệu thật có thiếu sót.
    """
    v_scaler = StandardScaler() if scale_v == "standard" else RobustScaler()
    return ColumnTransformer(
        transformers=[
            ("v", Pipeline([("imp", SimpleImputer(strategy="median")),
                            ("sc", v_scaler)]), V_AND_INTERACTION),
            ("amt_time", Pipeline([("imp", SimpleImputer(strategy="median")),
                                   ("sc", RobustScaler())]), AMOUNT_TIME_NUM),
            ("flag", SimpleImputer(strategy="most_frequent"), BINARY_COLS),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore",
                                      sparse_output=False)),
            ]), CATEGORICAL_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(model, sampler=None, feature_engineering: bool = True):
    """Pipeline hoàn chỉnh: FE -> preprocessing -> [sampler] -> model.

    Parameters
    ----------
    model : estimator
        Mô hình phân loại (có ``predict_proba``).
    sampler : imblearn sampler hoặc None
        SMOTE / RandomUnderSampler... Sampler của imblearn chỉ áp dụng khi
        ``fit`` (trên dữ liệu train của fold), không áp dụng khi dự đoán.
    feature_engineering : bool
        ``False`` để tạo pipeline ablation chỉ dùng đặc trưng gốc.
    """
    if feature_engineering:
        steps = [("fe", FraudFeatureEngineer()),
                 ("prep", build_preprocessor())]
    else:
        steps = [("fe", RawFeatureSelector()),
                 ("prep", build_raw_preprocessor())]
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("model", model))
    return Pipeline(steps)


# --------------------------------------------------------------------------- #
# Pipeline "đặc trưng gốc" dùng cho thí nghiệm ablation
# --------------------------------------------------------------------------- #
class RawFeatureSelector(BaseEstimator, TransformerMixin):
    """Chỉ chọn lại V1..V28 và Amount (bỏ Time), không tạo đặc trưng mới."""

    def fit(self, X, y=None):
        FraudFeatureEngineer._check_columns(X)
        return self

    def transform(self, X):
        FraudFeatureEngineer._check_columns(X)
        return X[V_COLUMNS + ["Amount"]].copy()


def build_raw_preprocessor() -> ColumnTransformer:
    """Preprocessor tối giản cho đặc trưng gốc (V chuẩn hóa, Amount robust)."""
    return ColumnTransformer(
        transformers=[
            ("v", Pipeline([("imp", SimpleImputer(strategy="median")),
                            ("sc", StandardScaler())]), V_COLUMNS),
            ("amt", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", RobustScaler())]), ["Amount"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
