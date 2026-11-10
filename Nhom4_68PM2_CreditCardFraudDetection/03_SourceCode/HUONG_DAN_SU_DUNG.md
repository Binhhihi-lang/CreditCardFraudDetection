# HƯỚNG DẪN CÀI ĐẶT & SỬ DỤNG MÃ NGUỒN (CHẠY THỰC TẾ & DOCKER)
## ĐỀ TÀI 6: PHÁT HIỆN GIAO DỊCH GIAN LẬN THẺ TÍN DỤNG (CREDIT CARD FRAUD DETECTION)
**Học phần:** Học máy (Machine Learning) - Mã HP: 608807  
**Khoa:** Công nghệ Thông tin - Trường Đại học Xây dựng Hà Nội (HUCE)  
**Nhóm thực hiện:** Nhóm 4 - Lớp: 68PM2  
**Thành viên:** Bình (Trưởng nhóm), Đức Anh, Hà Duy, Hoàng, Trần Duy  

---

## 1. TỔNG QUAN CẤU TRÚC THƯ MỤC NỘP BÀI

Gói sản phẩm hoàn chỉnh tuân thủ 100% quy định Rubric của Khoa CNTT - HUCE:

```
Nhom4_68PM2_CreditCardFraudDetection/
├── PHAN_CONG_CONG_VIEC.md            # Bảng phân công chi tiết & phao câu hỏi vấn đáp cá nhân
├── 01_Report/
│   └── Bao_cao_BTL_Nhom4.pdf         # Báo cáo kỹ thuật (15 - 25 trang A4 theo đúng 5 chương)
├── 02_Slide/
│   └── Slide_Thuyet_trinh_Nhom4.pdf  # Slide trình chiếu thuyết trình (12 - 15 trang)
├── 04_Video_Demo/
│   └── Video_Demo_Nhom4.mp4          # Video ngắn 3 phút chạy demo dự phòng
└── 03_SourceCode/                    # Toàn bộ mã nguồn dự án
    ├── data/
    │   ├── README.md                 # Hướng dẫn tải dữ liệu Kaggle
    │   ├── sample_transactions.csv   # 20 giao dịch mẫu (gồm cả ca gian lận thật để test)
    │   └── creditcard.csv            # (Tự tải từ Kaggle ~150MB, không đẩy lên Git)
    ├── notebooks/
    │   ├── 01_EDA_Preprocessing.ipynb    # Khám phá dữ liệu, làm sạch, kỹ nghệ đặc trưng
    │   └── 02_Modeling_Evaluation.ipynb  # Thử nghiệm 5 mô hình, tuning, SHAP, xuất model
    ├── models/
    │   └── best_model.joblib         # Pipeline hoàn chỉnh + Ngưỡng tối ưu + Metadata
    ├── reports/
    │   ├── figures/                  # 20 đồ thị biểu diễn kết quả (.png)
    │   └── tables/                   # 32 bảng số liệu định lượng (.csv)
    ├── fraud_pipeline.py             # Pipeline tiền xử lý & FraudFeatureEngineer
    ├── model_zoo.py                  # Khai báo 5 họ mô hình & không gian siêu tham số
    ├── evaluation.py                 # Hàm tính toán đa chỉ số, chọn ngưỡng, bootstrap CI
    ├── predict.py                    # Script nhận diện / suy luận giao dịch mới (CLI & hàm)
    ├── app.py                        # Webapp Live Demo tương tác bằng Streamlit (Đạt Mức 4)
    ├── Dockerfile                    # File đóng gói môi trường Docker
    ├── docker-compose.yml            # Khởi chạy ứng dụng 1 câu lệnh qua Docker Compose
    ├── .dockerignore
    ├── requirements.txt              # Danh sách thư viện cần thiết
    └── README.md
```

---

## 2. LỰA CHỌN CÁCH CHẠY DỰ ÁN

Để đảm bảo mã nguồn **chạy được trên mọi máy tính khác nhau** của các thành viên trong nhóm và máy tính của thầy cô mà không gặp bất kỳ lỗi xung đột môi trường nào, dự án hỗ trợ **2 phương thức triển khai**:

* **Cách 1 (Khuyên dùng để Demo): Chạy bằng Docker & Docker Compose** (Chỉ cần 1 câu lệnh, tự động cô lập môi trường, không cần cài Python).
* **Cách 2: Chạy trực tiếp trên máy bằng Python Virtual Environment (`.venv`)** (Dành cho máy đã cài sẵn Python 3.10 - 3.13).

---

## 3. CÁCH 1: CHẠY BẰNG DOCKER & DOCKER COMPOSE (ĐỒNG BỘ MỌI HỆ ĐIỀU HÀNH)

### Yêu cầu tiên quyết:
Máy tính đã cài sẵn **Docker Desktop** (hoặc Docker Engine trên Linux).

### Bước 1: Mở Terminal tại thư mục `03_SourceCode`
```bash
cd "Nhom4_68PM2_CreditCardFraudDetection/03_SourceCode"
```

### Bước 2: Khởi động Webapp Streamlit bằng Docker Compose
Chạy câu lệnh sau:
```bash
docker compose up --build -d webapp
```
* Docker sẽ tự động tải image Python 3.12, cài đặt toàn bộ thư viện trong `requirements.txt` và khởi chạy máy chủ Streamlit.
* Sau khi hoàn tất (mất khoảng 1-2 phút ở lần đầu tiên), mở trình duyệt web và truy cập:  
  👉 **`http://localhost:8501`**

### Bước 3: Xem hoặc chỉnh sửa Jupyter Notebook trong Docker (Tùy chọn)
Nếu muốn mở Jupyter Lab bên trong container Docker:
```bash
docker compose --profile jupyter up -d --build jupyter
```
Truy cập: 👉 **`http://localhost:8888`**

> [!NOTE]
> `jupyterlab` đã được thêm vào `requirements.txt`. Nếu trước đây bạn từng build image mà chưa có gói này, cần thêm cờ
> `--build` (như lệnh trên) để cài lại; thiếu nó container sẽ báo `Jupyter command jupyter-lab not found` rồi thoát.

### Bước 4: Chạy lệnh dự đoán bằng CLI trong Docker
```bash
docker compose exec webapp python predict.py --input data/sample_transactions.csv
```

### Dừng các container Docker:
```bash
docker compose down
```

---

## 4. CÁCH 2: CHẠY TRỰC TIẾP TRÊN MÁY BẰNG PYTHON VIRTUAL ENVIRONMENT

### Yêu cầu:
Python phiên bản từ **3.10 đến 3.13** (khuyến nghị Python 3.12).

### Bước 1: Mở Terminal tại thư mục `03_SourceCode`
```bash
cd "Nhom4_68PM2_CreditCardFraudDetection/03_SourceCode"
```

### Bước 2: Tạo môi trường ảo và kích hoạt
* **Trên Windows:**
  ```powershell
  python -m venv .venv
  .venv\Scripts\activate
  ```
* **Trên macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
*(Sau khi kích hoạt, đầu dòng lệnh sẽ xuất hiện `(.venv)`).*

### Bước 3: Cài đặt các thư viện cần thiết
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. HƯỚNG DẪN SỬ DỤNG CÁC CHỨC NĂNG

### A. Khởi chạy Ứng dụng Webapp Live Demo (Streamlit)
> [!TIP]
> Đây là sản phẩm quan trọng nhất giúp Nhóm 4 đạt điểm **Mức 4 (Xuất sắc: 85 - 100%)** ở phần Thuyết trình & Live Demo theo Rubric.

Chạy lệnh:
```bash
streamlit run app.py
```
Trình duyệt sẽ tự động mở giao diện ứng dụng tại `http://localhost:8501`. Ứng dụng cung cấp 4 chức năng:
1. **Kiểm tra Giao dịch Đơn lẻ:** Tự nhập số tiền, thời gian hoặc bấm **"Nạp mẫu Giao dịch Gian lận"** để xem hệ thống cảnh báo đỏ và khuyến nghị nghiệp vụ ngân hàng.
2. **Quét Hàng loạt (Batch Scanning):** Tải file CSV lên hoặc bấm **"Dùng file mẫu 20 giao dịch"**, hệ thống sẽ quét hàng loạt, tính toán tỷ lệ phát hiện và cho phép tải về file `predictions.csv`.
3. **Báo cáo Đa Mô hình:** Xem các bảng so sánh chỉ số CV và Test độc lập, đường cong PR/ROC và ma trận nhầm lẫn.
4. **Giải thích Mô hình (XAI & SHAP):** Trực quan hóa mức độ ảnh hưởng của từng đặc trưng ($V_4, V_{14}, V_{12}, V_{14}\times V_{10}$...) tới quyết định của thuật toán.

---

### B. Chạy kiểm tra bằng dòng lệnh (CLI `predict.py`)

Mô hình đã được lưu sẵn trong `models/best_model.joblib`. Bạn có thể kiểm tra dự đoán ngay mà không cần chạy lại quá trình huấn luyện:

1. **Dự đoán cho file giao dịch mẫu:**
   ```bash
   python predict.py --input data/sample_transactions.csv
   ```
   *Kết quả mong đợi:*
   ```text
   Mô hình: XGBoost | ngưỡng = 0.4671
   Đã phân tích 20 giao dịch, phát hiện 4 giao dịch nghi gian lận.
   (Có nhãn Class trong file: bắt được 4/5 giao dịch gian lận thật.)
   ```

2. **Dự đoán và xuất kết quả ra file CSV:**
   ```bash
   python predict.py --input data/sample_transactions.csv --output ket_qua.csv
   ```

3. **Thay đổi ngưỡng quyết định (Threshold):**
   ```bash
   python predict.py --input data/sample_transactions.csv --threshold 0.3
   ```

4. **Sử dụng trực tiếp trong mã Python:**
   ```python
   import pandas as pd
   from predict import predict

   df = pd.read_csv("data/sample_transactions.csv")
   result = predict(df) # Thêm cột fraud_probability và is_fraud
   print(result[["Amount", "fraud_probability", "is_fraud"]])
   ```

---

### C. Xem và Chạy lại Notebooks (Tùy chọn)

Cả hai notebook `01_EDA_Preprocessing.ipynb` và `02_Modeling_Evaluation.ipynb` **đã được chạy sẵn và lưu đầy đủ kết quả, đồ thị và bảng số liệu**. Nhóm có thể mở ra để xem ngay trong VS Code hoặc Jupyter Lab mà **không cần bấm chạy lại**.

Nếu muốn tự chạy lại từ đầu để kiểm chứng:
1. Tải file `creditcard.csv` từ Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Đặt file `creditcard.csv` vào thư mục `03_SourceCode/data/`.
3. Mở notebook trong VS Code, chọn Kernel là `.venv`.
4. Chạy tuần tự:
   - Chạy `01_EDA_Preprocessing.ipynb` (~2 phút).
   - Chạy `02_Modeling_Evaluation.ipynb` (quá trình huấn luyện 5 mô hình và tinh chỉnh siêu tham số mất khoảng 1.5 - 2 giờ tùy cấu hình CPU).

---

## 6. LƯU Ý QUAN TRỌNG KHI NỘP BÀI & BẢO VỆ

1. **Quy định file nộp lên LMS:**
   - **KHÔNG NỘP** thư mục `.venv`, `__pycache__` và file dữ liệu gốc `creditcard.csv` (vì dung lượng 150MB quá nặng, hệ thống LMS sẽ chặn).
   - Chỉ nộp file `sample_transactions.csv` trong thư mục `data/`.
2. **Quy tắc Chống Rò rỉ Dữ liệu (Tránh bị trừ 1.0 - 2.0 điểm):**
   - Đảm bảo trả lời đúng câu hỏi vấn đáp: Nhóm chia Train/Test **trước** khi tiền xử lý và mọi bộ scaler/imputer/SMOTE đều nằm **bên trong Pipeline**.
3. **Quy định Đạo code (Tránh bị điểm 0):**
   - Từng thành viên học thuộc kỹ bảng câu hỏi vấn đáp cá nhân trong file `PHAN_CONG_CONG_VIEC.md`.

