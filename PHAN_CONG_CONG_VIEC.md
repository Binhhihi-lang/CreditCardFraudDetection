# KẾ HOẠCH PHÂN CÔNG CÔNG VIỆC & HƯỚNG DẪN VẤN ĐÁP BTL HỌC MÁY
## ĐỀ TÀI 6: PHÁT HIỆN GIAO DỊCH GIAN LẬN THẺ TÍN DỤNG (CREDIT CARD FRAUD DETECTION)
**Học phần:** Học máy (Machine Learning) - Mã HP: 608807  
**Khoa:** Công nghệ Thông tin - Trường Đại học Xây dựng Hà Nội (HUCE)  
**Nhóm thực hiện:** Nhóm 4 - Lớp: 68PM2  

---

> Chi tiết đầy đủ xem tại: [`Nhom4_68PM2_CreditCardFraudDetection/PHAN_CONG_CONG_VIEC.md`](Nhom4_68PM2_CreditCardFraudDetection/PHAN_CONG_CONG_VIEC.md)

### BẢNG PHÂN CÔNG NHIỆM VỤ NHANH

| STT | Thành viên | Mảng phụ trách | Nhiệm vụ kỹ thuật | Tỷ lệ |
|:---:|:---|:---|:---|:---:|
| **1** | **Bình** *(Nhóm trưởng)* | **Kiến trúc Pipeline, XGBoost, Webapp Streamlit & Triển khai Docker** | Thiết kế Pipeline tổng thể, huấn luyện XGBoost, tối ưu ngưỡng quyết định (Threshold Tuning), đóng gói mô hình, phát triển Webapp Streamlit và Docker hóa. | **20%** |
| **2** | **Đức Anh** | **Khám phá Dữ liệu (EDA), Tiền xử lý & Kỹ nghệ Đặc trưng** | Chia tập dữ liệu chống rò rỉ (80/20 Stratified), phân tích EDA chuyên sâu, lập trình `FraudFeatureEngineer` (tạo `Amount_log`, `Hour`, `Period`, `V_norm`...), cấu hình `ColumnTransformer`. | **20%** |
| **3** | **Hà Duy** | **Kỹ thuật Cân bằng Dữ liệu & Mô hình Tuyến tính / Cơ bản** | Thử nghiệm 3 chiến lược (Class Weight, SMOTE, Undersampling), xây dựng Baseline DummyClassifier, huấn luyện và tối ưu Logistic Regression & k-NN bằng GridSearchCV. | **20%** |
| **4** | **Hoàng** | **Mô hình Dựa trên Cây (Decision Tree, Random Forest) & Cross-Validation** | Triển khai Decision Tree, Random Forest, thiết lập 5-Fold Stratified Cross-Validation, tinh chỉnh siêu tham số bằng RandomizedSearchCV, vẽ và phân tích Learning Curve. | **20%** |
| **5** | **Trần Duy** | **Đánh giá Chuyên sâu, Phân tích Lỗi, Giải thích Mô hình (SHAP) & Báo cáo BTL** | So sánh đa chỉ số (PR-AUC, ROC-AUC, Recall, F2, MCC), phân tích ma trận nhầm lẫn & các ca False Negative, vẽ SHAP Summary Plot, tổng hợp Báo cáo kỹ thuật (15-25 trang) và Slide. | **20%** |

