"""Hàm hỗ trợ đánh giá mô hình cho bài toán phân loại mất cân bằng.

Gồm: bộ chỉ số đa chiều, chọn ngưỡng quyết định, khoảng tin cậy bootstrap và
các hàm vẽ biểu đồ dùng chung cho notebook 02.

Nguyên tắc: ngưỡng quyết định chỉ được chọn từ dự đoán out-of-fold của tập
train, tuyệt đối không dùng tập test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    make_scorer,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Bộ scorer dùng cho cross_validate: PR-AUC là chỉ số chính, Recall là chỉ số
# thứ hai theo yêu cầu đề tài.
CV_SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "recall": "recall",
    "precision": make_scorer(precision_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "f2": make_scorer(fbeta_score, beta=2, zero_division=0),
}


def compute_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    """Tính các chỉ số đánh giá tại một ngưỡng quyết định.

    ``y_score`` là xác suất lớp dương. PR-AUC (average precision) và ROC-AUC
    không phụ thuộc ngưỡng; các chỉ số còn lại tính tại ``threshold``.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "PR-AUC": average_precision_score(y_true, y_score),
        "ROC-AUC": roc_auc_score(y_true, y_score),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "F2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "Threshold": float(threshold),
    }


def choose_threshold(y_true, y_score, objective: str = "f2",
                     min_recall: float = 0.85) -> float:
    """Chọn ngưỡng quyết định từ dự đoán out-of-fold.

    objective:
        * ``"f2"``: cực đại hóa F2 (Recall nặng gấp đôi Precision).
        * ``"recall_floor"``: ngưỡng cho Precision cao nhất mà vẫn đạt
          ``Recall >= min_recall``.
        * ``"f1"``: cực đại hóa F1.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision/recall có thêm một phần tử cuối (không có ngưỡng tương ứng).
    precision, recall = precision[:-1], recall[:-1]
    if objective in ("f1", "f2"):
        beta2 = 1.0 if objective == "f1" else 4.0
        denom = beta2 * precision + recall
        score = np.divide((1 + beta2) * precision * recall, denom,
                          out=np.zeros_like(denom), where=denom > 0)
        # Nếu nhiều ngưỡng cùng điểm, lấy ngưỡng nhỏ nhất ổn định theo thứ tự.
        return float(thresholds[int(np.argmax(score))])
    if objective == "recall_floor":
        ok = recall >= min_recall
        if not ok.any():
            return float(thresholds[0])
        idx = np.where(ok)[0]
        return float(thresholds[idx[np.argmax(precision[idx])]])
    raise ValueError(f"objective không hợp lệ: {objective}")


def bootstrap_ci(y_true, y_score, threshold: float, n_boot: int = 1000,
                 seed: int = 42, alpha: float = 0.05) -> pd.DataFrame:
    """Khoảng tin cậy bootstrap (phân tầng theo lớp) cho PR-AUC, Recall, Precision.

    Tập test chỉ có vài chục giao dịch gian lận nên các chỉ số dao động mạnh;
    khoảng tin cậy cho biết độ bất định đó.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    rows = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        m = compute_metrics(y_true[idx], y_score[idx], threshold)
        rows.append((m["PR-AUC"], m["Recall"], m["Precision"], m["F2"]))
    arr = np.array(rows)
    lo, hi = np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    return pd.DataFrame({"Chỉ số": ["PR-AUC", "Recall", "Precision", "F2"],
                         "Cận dưới 95%": lo, "Cận trên 95%": hi})
