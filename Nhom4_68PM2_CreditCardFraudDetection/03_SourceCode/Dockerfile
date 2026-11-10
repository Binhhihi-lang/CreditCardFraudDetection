# Sử dụng Python 3.12 slim để tối ưu dung lượng image và đảm bảo tính tương thích cao
FROM python:3.12-slim

# Thiết lập biến môi trường
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Cài đặt các gói hệ thống cần thiết (libgomp1 cần cho XGBoost và OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép file requirements.txt và cài đặt thư viện
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Sao chép toàn bộ mã nguồn vào container
COPY . .

# Mở cổng cho Webapp Streamlit (8501) và Jupyter Notebook (8888)
EXPOSE 8501 8888

# Mặc định khởi động ứng dụng Webapp Streamlit phục vụ Live Demo
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

