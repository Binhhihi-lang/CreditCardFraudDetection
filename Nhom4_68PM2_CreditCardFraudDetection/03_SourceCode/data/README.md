# Dữ liệu

Notebook và `predict.py` đọc file **`data/creditcard.csv`**.

## Nguồn dữ liệu
* Kaggle – *Credit Card Fraud Detection*: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
* Tác giả: Machine Learning Group, Université Libre de Bruxelles (ULB), phối hợp với Worldline.
* Nội dung: 284.807 giao dịch thẻ tín dụng của khách hàng châu Âu trong hai ngày (tháng 9/2013), trong đó 492 giao dịch gian lận (≈ 0,172%).

## Cách lấy dữ liệu
1. Đăng nhập Kaggle, mở trang dữ liệu ở trên, bấm **Download** (file `creditcardfraud.zip`).
2. Giải nén, đặt file `creditcard.csv` (≈ 150 MB) vào thư mục này: `03_SourceCode/data/creditcard.csv`.

Hoặc dùng Kaggle CLI:
```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip
```

## Mô tả cột
| Cột | Ý nghĩa |
|---|---|
| `Time` | Số giây kể từ giao dịch đầu tiên trong bộ dữ liệu |
| `V1` ... `V28` | Các thành phần chính (PCA) của dữ liệu gốc, đã ẩn danh vì bảo mật |
| `Amount` | Số tiền giao dịch |
| `Class` | Nhãn: 1 = gian lận, 0 = bình thường |

## File khác trong thư mục
* `sample_transactions.csv`: 20 giao dịch mẫu (5 gian lận, 15 bình thường, lấy từ tập test) dùng để thử `predict.py`. Chỉ để minh họa.

## Trích dẫn
Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). *Calibrating Probability with Undersampling for Unbalanced Classification.* IEEE Symposium Series on Computational Intelligence (SSCI).
