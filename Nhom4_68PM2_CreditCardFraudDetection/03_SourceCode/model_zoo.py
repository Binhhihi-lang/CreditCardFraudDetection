"""Khai báo các họ mô hình, chiến lược xử lý mất cân bằng và không gian siêu tham số.

Mọi mô hình đều được bọc trong cùng một Pipeline (xem ``fraud_pipeline.build_pipeline``)
nên chỉ cần thay ``family`` và ``strategy`` là có ngay pipeline đầy đủ:

    pipe = make_pipeline("XGBoost", "class_weight", scale_pos_weight=599)

Tên bước trong pipeline: ``fe`` -> ``prep`` -> [``sampler``] -> ``model``.
Vì vậy tên siêu tham số khi tinh chỉnh có dạng ``model__max_depth`` hoặc
``sampler__sampling_strategy``.
"""

from __future__ import annotations

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from fraud_pipeline import RANDOM_STATE, build_pipeline

FAMILIES = ["Logistic Regression", "k-NN", "Decision Tree", "Random Forest", "XGBoost"]
STRATEGIES = ("class_weight", "SMOTE", "Undersampling")
# Tỷ lệ (lớp thiểu số / lớp đa số) sau khi lấy mẫu lại; 0.1 nghĩa là 1 gian lận : 10 bình thường.
SAMPLING_RATIO = 0.1


def valid_strategies(family: str) -> tuple[str, ...]:
    """k-NN không có tham số trọng số lớp; SMOTE làm k-NN rất chậm (tìm lân cận
    trên hàng trăm nghìn mẫu ở 43 chiều) nên chỉ thử undersampling."""
    return ("Undersampling",) if family == "k-NN" else STRATEGIES


def make_sampler(strategy: str, ratio: float = SAMPLING_RATIO):
    """Tạo sampler của imbalanced-learn (hoặc None nếu dùng trọng số lớp)."""
    if strategy == "SMOTE":
        return SMOTE(sampling_strategy=ratio, random_state=RANDOM_STATE)
    if strategy == "Undersampling":
        return RandomUnderSampler(sampling_strategy=ratio, random_state=RANDOM_STATE)
    return None


def make_model(family: str, strategy: str, scale_pos_weight: float = 1.0, n_jobs: int = 1):
    """Mô hình với siêu tham số khởi đầu hợp lý (sẽ được tinh chỉnh sau)."""
    use_cw = strategy == "class_weight"
    if family == "Logistic Regression":
        return LogisticRegression(C=1.0, class_weight="balanced" if use_cw else None,
                                  max_iter=3000)
    if family == "k-NN":
        return KNeighborsClassifier(n_neighbors=5, n_jobs=n_jobs)
    if family == "Decision Tree":
        return DecisionTreeClassifier(max_depth=8, min_samples_leaf=5,
                                      class_weight="balanced" if use_cw else None,
                                      random_state=RANDOM_STATE)
    if family == "Random Forest":
        return RandomForestClassifier(
            n_estimators=150, max_depth=14, min_samples_leaf=2, max_samples=0.3,
            class_weight="balanced_subsample" if use_cw else None,
            n_jobs=n_jobs, random_state=RANDOM_STATE)
    if family == "XGBoost":
        return XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=scale_pos_weight if use_cw else 1.0,
            tree_method="hist", eval_metric="aucpr", n_jobs=n_jobs,
            random_state=RANDOM_STATE)
    raise ValueError(f"Họ mô hình không hợp lệ: {family}")


def make_pipeline(family: str, strategy: str, scale_pos_weight: float = 1.0,
                  n_jobs: int = 1, feature_engineering: bool = True):
    """Pipeline đầy đủ cho một cặp (họ mô hình, chiến lược mất cân bằng)."""
    return build_pipeline(
        make_model(family, strategy, scale_pos_weight, n_jobs),
        sampler=make_sampler(strategy),
        feature_engineering=feature_engineering,
    )


def make_baseline():
    """Baseline ngây thơ: luôn trả về tỷ lệ lớp thiểu số (không dùng thông tin đặc trưng)."""
    return build_pipeline(DummyClassifier(strategy="prior"))


def param_space(family: str, strategy: str, scale_pos_weight: float):
    """Không gian tìm kiếm siêu tham số cho một họ mô hình.

    Trả về (``param_grid``, ``search``) với ``search`` là ``"grid"`` hoặc
    ``"random"``. Mô hình đơn giản/ít tham số dùng GridSearchCV; mô hình
    ensemble (nhiều tham số, huấn luyện lâu) dùng RandomizedSearchCV.
    """
    space: dict = {}
    # Tham số của sampler (nếu có)
    if strategy == "SMOTE":
        space["sampler__sampling_strategy"] = [0.05, 0.1, 0.3]
    elif strategy == "Undersampling":
        space["sampler__sampling_strategy"] = [0.02, 0.05, 0.1, 0.3]

    use_cw = strategy == "class_weight"
    cw_options = ["balanced", {0: 1, 1: 10}, {0: 1, 1: 50}]

    if family == "Logistic Regression":
        space["model__C"] = [0.001, 0.01, 0.1, 1, 10, 100]
        if use_cw:
            space["model__class_weight"] = cw_options
        return space, "grid"
    if family == "k-NN":
        space["model__n_neighbors"] = [3, 5, 11, 21, 41]
        space["model__weights"] = ["uniform", "distance"]
        return space, "grid"
    if family == "Decision Tree":
        space["model__max_depth"] = [4, 6, 8, 12]
        space["model__min_samples_leaf"] = [5, 20, 50]
        space["model__criterion"] = ["gini", "entropy"]
        if use_cw:
            space["model__class_weight"] = cw_options
        return space, "random"
    if family == "Random Forest":
        space["model__n_estimators"] = [100, 200, 300]
        space["model__max_depth"] = [8, 12, 16]
        space["model__min_samples_leaf"] = [1, 2, 5, 10]
        space["model__max_features"] = ["sqrt", 0.3]
        space["model__max_samples"] = [0.1, 0.2, 0.4]
        if use_cw:
            space["model__class_weight"] = ["balanced_subsample", {0: 1, 1: 10}, {0: 1, 1: 50}]
        return space, "random"
    if family == "XGBoost":
        space["model__n_estimators"] = [200, 300, 500]
        space["model__max_depth"] = [3, 4, 5, 6, 8]
        space["model__learning_rate"] = [0.02, 0.05, 0.1, 0.2]
        space["model__subsample"] = [0.7, 0.85, 1.0]
        space["model__colsample_bytree"] = [0.6, 0.8, 1.0]
        space["model__min_child_weight"] = [1, 3, 5]
        space["model__gamma"] = [0, 0.5, 1]
        space["model__reg_lambda"] = [1, 5, 10]
        if use_cw:
            space["model__scale_pos_weight"] = [1, 10, 50, round(scale_pos_weight)]
        return space, "random"
    raise ValueError(f"Họ mô hình không hợp lệ: {family}")
