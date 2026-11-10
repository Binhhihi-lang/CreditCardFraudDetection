# HƯỚNG DẪN CÀI ĐẶT & SỬ DỤNG MÃ NGUỒN (CHẠY THỰC TẾ & DOCKER)
## ĐỀ TÀI 6: PHÁT HIỆN GIAO DỊCH GIAN LẬN THẺ TÍN DỤNG (CREDIT CARD FRAUD DETECTION)
**Học phần:** Học máy (Machine Learning) - Mã HP: 608807  
**Khoa:** Công nghệ Thông tin - Trường Đại học Xây dựng Hà Nội (HUCE)  
**Nhóm thực hiện:** Nhóm 4 - Lớp: 68PM2  
**Thành viên:** Bình (Trưởng nhóm), Đức Anh, Hà Duy, Hoàng, Trần Duy  

---

> Chi tiết đầy đủ xem tại: [`Nhom4_68PM2_CreditCardFraudDetection/03_SourceCode/HUONG_DAN_SU_DUNG.md`](Nhom4_68PM2_CreditCardFraudDetection/03_SourceCode/HUONG_DAN_SU_DUNG.md)

### TÓM TẮT CÁC LỆNH CHẠY NHANH

#### Cách 1: Chạy bằng Docker (Khuyên dùng để đồng bộ mọi máy & Demo)
```bash
cd "Nhom4_68PM2_CreditCardFraudDetection/03_SourceCode"
docker compose up --build -d webapp
```
👉 Mở trình duyệt truy cập: **`http://localhost:8501`** để trải nghiệm ứng dụng Webapp Live Demo.

#### Cách 2: Chạy trực tiếp bằng Python
```powershell
cd "Nhom4_68PM2_CreditCardFraudDetection/03_SourceCode"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Chạy Webapp Demo:
streamlit run app.py

# Hoặc chạy kiểm tra CLI:
python predict.py --input data/sample_transactions.csv
```

