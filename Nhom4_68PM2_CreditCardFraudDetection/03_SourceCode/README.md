# Phát hiện giao dịch gian lận thẻ tín dụng (Credit Card Fraud Detection)

**Học phần:** Học máy (Machine Learning) - Mã HP: 608807  
**Khoa:** Công nghệ Thông tin - Trường Đại học Xây dựng Hà Nội (HUCE)  
**Nhóm thực hiện:** Nhóm 4 - Lớp: 68PM2  
**Thành viên:** Bình (Trưởng nhóm), Đức Anh, Hà Duy, Hoàng, Trần Duy  

Bài toán: phân loại **cực kỳ mất cân bằng** (≈ 0,17% giao dịch là gian lận). Chỉ số chính là **PR-AUC** và **Recall**
(kèm Precision, F2, MCC); *không* dùng Accuracy làm thước đo vì mô hình "luôn đoán bình thường" đã đạt 99,83%.

---

## 1. Cấu trúc thư mục

```
03_SourceCode/
├── data/
│   ├── README.md                  # nguồn dữ liệu, cách tải, mô tả cột
│   ├── creditcard.csv             # (tự tải về từ Kaggle, không nằm trong gói nộp vì ~150 MB)
│   └── sample_transactions.csv    # 20 giao dịch mẫu để thử predict.py & app.py
├── notebooks/
│   ├── 01_EDA_Preprocessing.ipynb     # EDA, làm sạch, chia train/test, kỹ nghệ đặc trưng
│   └── 02_Modeling_Evaluation.ipynb   # mô hình, CV, tuning, đánh giá, phân tích lỗi, xuất mô hình
├── models/best_model.joblib       # pipeline hoàn chỉnh + ngưỡng + siêu dữ liệu
├── reports/figures/, reports/tables/  # hình và bảng do notebook sinh ra (dùng cho báo cáo/slide)
├── fraud_pipeline.py              # nạp/làm sạch/chia dữ liệu, FraudFeatureEngineer, Pipeline + ColumnTransformer
├── model_zoo.py                   # 5 họ mô hình, chiến lược mất cân bằng, không gian siêu tham số
├── evaluation.py                  # các chỉ số, chọn ngưỡng, bootstrap khoảng tin cậy
├── predict.py                     # dự đoán cho giao dịch mới (CLI và hàm Python)
├── app.py                         # Webapp Live Demo tương tác bằng Streamlit (Đạt chuẩn Mức 4 Rubric)
├── Dockerfile                     # Docker image đóng gói môi trường chuẩn
├── docker-compose.yml             # Khởi chạy một lệnh cho Webapp và Jupyter
├── requirements.txt               # Thư viện Python cần thiết
└── setup.cfg                      # cấu hình flake8
```

---

## 2. Hướng dẫn Cài đặt & Khởi chạy

### Cách 1: Chạy bằng Docker (Khuyên dùng - Đảm bảo chạy trên mọi máy tính)
```bash
docker compose up --build -d webapp
```
Truy cập Webapp Live Demo: 👉 **`http://localhost:8501`**

### Cách 2: Chạy trực tiếp bằng Python
```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Khởi chạy Webapp:
```bash
streamlit run app.py
```

---

## 3. Dự đoán giao dịch mới (`predict.py`)

```bash
python predict.py --input data/sample_transactions.csv --output predictions.csv
python predict.py --input data/sample_transactions.csv --threshold 0.3   # đổi ngưỡng quyết định
```

Dùng trong mã Python:
```python
import pandas as pd
from predict import predict

df = pd.read_csv("data/sample_transactions.csv")
result = predict(df) # Trả về DataFrame kèm 2 cột: fraud_probability và is_fraud
print(result[["Amount", "fraud_probability", "is_fraud"]])
```

---

## 4. Thiết kế chống rò rỉ dữ liệu (Anti-Data Leakage)

* Loại dòng trùng và **chia train/test (80/20, phân tầng) ngay từ đầu**, trước khi EDA; tập test chỉ dùng **một lần** ở mục 9 của notebook 02.
* Toàn bộ imputer, scaler, encoder và bộ lấy mẫu lại (SMOTE / undersampling) nằm **bên trong** `imblearn.pipeline.Pipeline`, nên được fit lại trong từng fold của CV và không bao giờ thấy dữ liệu validation/test.
* `FraudFeatureEngineer` **không có trạng thái** (mỗi đặc trưng chỉ tính từ chính dòng đó).
* Ngưỡng quyết định chọn từ dự đoán out-of-fold của **riêng tập train** (ngưỡng tối ưu $p^* \approx 0.4671$).

---

## 5. Kết quả chính trên Test Set độc lập (56,746 giao dịch - 95 ca gian lận)

Số liệu lấy trực tiếp từ `reports/tables/test_comparison_all_models.csv`. Cột "PR-AUC (CV)" là trung bình 5-fold trên tập train; các cột còn lại tính trên tập test tại **ngưỡng tối ưu riêng của từng mô hình** (chọn bằng cực đại F2 trên dự đoán out-of-fold của train).

| Mô hình | PR-AUC (CV 5-fold) | PR-AUC (Test) | Recall (Test) | Precision (Test) | F2 (Test) | MCC (Test) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **XGBoost (Tối ưu)** | **0.8572 ± 0.0149** | **0.8197** | **0.768** | **0.961** | **0.800** | **0.859** |
| Random Forest | 0.8501 ± 0.0167 | 0.8194 | 0.758 | 0.878 | 0.779 | 0.815 |
| Logistic Regression | 0.8098 ± 0.0230 | 0.7601 | 0.779 | 0.787 | 0.781 | 0.783 |
| k-NN | 0.7642 ± 0.0444 | 0.7244 | 0.779 | 0.712 | 0.764 | 0.744 |
| Decision Tree | 0.7273 ± 0.0294 | 0.6859 | 0.716 | 0.895 | 0.746 | 0.800 |
| *Baseline (Dummy)* | 0.0017 | 0.0017 | 0.0 | 0.0 | 0.0 | 0.0 |

> **Lưu ý khi đọc bảng:** XGBoost và Random Forest gần như tương đương về PR-AUC (chênh 0,0003; bootstrap ghép cặp 95% CI
> [-0,016 ; +0,015], chỉ thắng 53% số lần) — xem `reports/tables/paired_bootstrap_vs_best.csv`. XGBoost được chọn vì
> PR-AUC trung bình CV cao nhất và thắng 4/5 fold, **không** phải vì vượt trội có ý nghĩa thống kê.
> Ở ngưỡng mặc định 0,5, XGBoost cho Recall 0,768 / Precision 0,961 / F2 0,800 (trùng với ngưỡng tối ưu 0,4671 trên mô hình này).

**Confusion matrix của XGBoost trên test (tại ngưỡng 0,4671):** TP = 73, FP = 3, FN = 22, TN = 56.648.
Khoảng tin cậy bootstrap 95% (1.000 lần): PR-AUC [0,739 ; 0,890], Recall [0,684 ; 0,853], Precision [0,915 ; 1,000].

---

## 6. Tài liệu đính kèm
* Kế hoạch phân công & Phao câu hỏi vấn đáp: [`../PHAN_CONG_CONG_VIEC.md`](../PHAN_CONG_CONG_VIEC.md)
* Hướng dẫn sử dụng chi tiết: [`HUONG_DAN_SU_DUNG.md`](HUONG_DAN_SU_DUNG.md)
