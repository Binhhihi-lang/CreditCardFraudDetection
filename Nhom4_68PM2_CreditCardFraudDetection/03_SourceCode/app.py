"""Ứng dụng Webapp Phát hiện Giao dịch Gian lận Thẻ Tín dụng.

BTL Môn Học máy (Machine Learning) - Mã HP: 608807
Khoa Công nghệ Thông tin - Trường Đại học Xây dựng Hà Nội (HUCE)
Nhóm thực hiện: Nhóm 4 - Lớp: 68PM2
Thành viên: Bình (Trưởng nhóm), Đức Anh, Hà Duy, Hoàng, Trần Duy

Chạy ứng dụng:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Credit Card Fraud Detection - Nhóm 4 (68PM2)",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import các hàm nghiệp vụ từ predict.py và fraud_pipeline.py
import fraud_pipeline
from predict import MODEL_PATH, load_model, predict, RAW_FEATURES

# --------------------------------------------------------------------------- #
# Giao diện Sidebar & Thông tin Nhóm
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/Huce_logo.png/640px-Huce_logo.png",
        width=110,
    )
    st.title("💳 Fraud Detection")
    st.markdown("**BTL Học máy (HUCE - 608807)**")
    st.markdown("**Nhóm 4 - Lớp: 68PM2**")
    st.markdown("---")
    st.markdown("### 👥 Thành viên nhóm:")
    st.markdown(
        """
        1. **Bình (Trưởng nhóm)**: Pipeline & XGBoost, App
        2. **Đức Anh**: EDA & Kỹ nghệ đặc trưng
        3. **Hà Duy**: Xử lý mất cân bằng & Mô hình tuyến tính
        4. **Hoàng**: Cây quyết định, Random Forest & CV
        5. **Trần Duy**: Đánh giá đa chỉ số, SHAP & Báo cáo
        """
    )
    st.markdown("---")

    # Tải mô hình để lấy ngưỡng tối ưu
    try:
        model_bundle = load_model(MODEL_PATH)
        default_threshold = float(model_bundle.get("threshold", 0.4671))
        st.success(f" Đã nạp mô hình: **{model_bundle.get('model_name', 'XGBoost')}**")
    except Exception as e:
        model_bundle = None
        default_threshold = 0.4671
        st.warning(f"⚠️ Chưa nạp được mô hình từ file: {e}")

    st.markdown("### ⚙️ Cấu hình Ngưỡng quyết định")
    threshold = st.slider(
        "Ngưỡng phân loại gian lận (Threshold $p^*$):",
        min_value=0.01,
        max_value=0.99,
        value=default_threshold,
        step=0.01,
        help="Giao dịch có xác suất >= ngưỡng này sẽ bị gắn cờ là Gian lận (Fraud). Mặc định 0.4671 là ngưỡng tối ưu F2-score trên tập Train.",
    )
    st.caption(f"Ngưỡng tối ưu lưu cùng mô hình: **{default_threshold:.4f}**")

# --------------------------------------------------------------------------- #
# Tiêu đề chính
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <div style='background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 22px; border-radius: 10px; color: white; margin-bottom: 25px;'>
        <h1 style='margin:0; font-size: 28px; color: #ffffff;'>HỆ THỐNG PHÁT HIỆN GIAO DỊCH GIAN LẬN THẺ TÍN DỤNG</h1>
        <p style='margin: 6px 0 0 0; font-size: 15px; opacity: 0.9;'>
            Ứng dụng Học máy Phân loại Mất cân bằng Cực độ (Imbalanced Classification) | Nhóm 4 - Lớp 68PM2 - ĐH Xây dựng Hà Nội (HUCE)
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Tabs chức năng
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔍 Kiểm tra Giao dịch Đơn lẻ",
        "📂 Quét Hàng loạt (File CSV)",
        "📊 Báo cáo Đa Mô hình & Hiệu năng",
        "🧠 Giải thích Mô hình (XAI & SHAP)",
    ]
)

# --------------------------------------------------------------------------- #
# TAB 1: KIỂM TRA GIAO DỊCH ĐƠN LẺ
# --------------------------------------------------------------------------- #
with tab1:
    st.subheader("Kiểm tra Rủi ro cho một Giao dịch Mới")
    st.write(
        "Nhập các thông số giao dịch hoặc chọn mẫu thử nghiệm có sẵn để kiểm tra hệ thống suy luận thời gian thực."
    )

    sample_csv_path = Path(__file__).resolve().parent / "data" / "sample_transactions.csv"
    sample_df = None
    if sample_csv_path.exists():
        sample_df = pd.read_csv(sample_csv_path)

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    with col_btn1:
        load_legit = st.button(" Nạp mẫu Giao dịch Hợp lệ")
    with col_btn2:
        load_fraud = st.button("🚨 Nạp mẫu Giao dịch Gian lận")

    # Mặc định khởi tạo giá trị
    initial_values = {f"V{i}": 0.0 for i in range(1, 29)}
    initial_values["Time"] = 45000.0
    initial_values["Amount"] = 125.50

    if load_legit and sample_df is not None:
        legit_row = sample_df[sample_df["Class"] == 0].iloc[0]
        for col in RAW_FEATURES:
            initial_values[col] = float(legit_row[col])
        st.info("Đã nạp mẫu giao dịch HỢP LỆ từ tập mẫu.")
    elif load_fraud and sample_df is not None:
        fraud_row = sample_df[sample_df["Class"] == 1].iloc[0]
        for col in RAW_FEATURES:
            initial_values[col] = float(fraud_row[col])
        st.warning("Đã nạp mẫu giao dịch GIAN LẬN thực tế từ tập mẫu.")

    with st.form("single_transaction_form"):
        col_amt, col_time = st.columns(2)
        with col_amt:
            amount_val = st.number_input(
                "Số tiền giao dịch (Amount - USD):",
                min_value=0.0,
                max_value=100000.0,
                value=float(initial_values["Amount"]),
                step=10.0,
            )
        with col_time:
            time_val = st.number_input(
                "Thời gian giao dịch (Time - giây kể từ giao dịch đầu):",
                min_value=0.0,
                max_value=200000.0,
                value=float(initial_values["Time"]),
                step=100.0,
                help=f"Tương đương giờ trong ngày: {(float(initial_values['Time']) % 86400) / 3600:.1f}h",
            )

        st.markdown("**Các biến PCA nhạy cảm nhất (Quyết định mức độ rủi ro):**")
        col_v1, col_v2, col_v3, col_v4 = st.columns(4)
        with col_v1:
            v14_val = st.number_input("V14 (Tương quan âm mạnh):", value=float(initial_values["V14"]))
            v12_val = st.number_input("V12 (Tương quan âm):", value=float(initial_values["V12"]))
        with col_v2:
            v10_val = st.number_input("V10 (Tương quan âm):", value=float(initial_values["V10"]))
            v17_val = st.number_input("V17 (Tương quan âm):", value=float(initial_values["V17"]))
        with col_v3:
            v4_val = st.number_input("V4 (Tương quan dương):", value=float(initial_values["V4"]))
            v11_val = st.number_input("V11 (Tương quan dương):", value=float(initial_values["V11"]))
        with col_v4:
            v2_val = st.number_input("V2:", value=float(initial_values["V2"]))
            v3_val = st.number_input("V3:", value=float(initial_values["V3"]))

        with st.expander("Hiển thị & Chỉnh sửa toàn bộ các biến V còn lại (V1, V5-V9, V13, V15-V16, V18-V28)"):
            other_cols = [c for c in [f"V{i}" for i in range(1, 29)] if c not in ["V14", "V12", "V10", "V17", "V4", "V11", "V2", "V3"]]
            sub_cols = st.columns(4)
            other_v_inputs = {}
            for idx, col_name in enumerate(other_cols):
                with sub_cols[idx % 4]:
                    other_v_inputs[col_name] = st.number_input(
                        col_name, value=float(initial_values[col_name]), key=f"inp_{col_name}"
                    )

        submitted = st.form_submit_button("🚀 Phân tích Rủi ro Giao dịch", use_container_width=True)

    if submitted:
        # Gom dữ liệu thành DataFrame 1 dòng
        row_dict = {
            "Time": time_val,
            "Amount": amount_val,
            "V14": v14_val,
            "V12": v12_val,
            "V10": v10_val,
            "V17": v17_val,
            "V4": v4_val,
            "V11": v11_val,
            "V2": v2_val,
            "V3": v3_val,
        }
        for k, v in other_v_inputs.items():
            row_dict[k] = v

        input_df = pd.DataFrame([row_dict])
        try:
            res_df = predict(input_df, threshold=threshold)
            prob = float(res_df["fraud_probability"].iloc[0])
            is_fraud = int(res_df["is_fraud"].iloc[0])

            st.markdown("---")
            st.markdown("### Kết quả Phân tích:")

            col_res1, col_res2 = st.columns([1, 2])
            with col_res1:
                st.metric(
                    label="Xác suất Gian lận (Fraud Probability)",
                    value=f"{prob * 100:.2f}%",
                    delta=f"Ngưỡng: {threshold * 100:.1f}%",
                    delta_color="inverse" if prob >= threshold else "normal",
                )
                st.progress(prob)

            with col_res2:
                if is_fraud == 1:
                    st.error(
                        f"""
                        🚨 **CẢNH BÁO: GIAO DỊCH CÓ DẤU HIỆU GIAN LẬN!**  
                        * Xác suất rủi ro **{prob * 100:.2f}%** vượt quá ngưỡng an toàn **{threshold * 100:.1f}%**.  
                        * **Khuyến nghị hệ thống Core Banking:**  
                          1. Lập tức tạm giữ giao dịch (Hold Transaction).  
                          2. Kích hoạt xác thực sinh trắc học hoặc OTP qua tin nhắn SMS/SmartOTP.  
                          3. Thông báo cho bộ phận Phòng chống Gian lận (Fraud Operations Unit).
                        """
                    )
                else:
                    st.success(
                        f"""
                         **GIAO DỊCH HỢP LỆ (AN TOÀN)**  
                        * Xác suất rủi ro **{prob * 100:.2f}%** nằm trong vùng kiểm soát an toàn (< {threshold * 100:.1f}%).  
                        * **Khuyến nghị hệ thống Core Banking:** Phê duyệt giao dịch (Auto-Approved).
                        """
                    )
        except Exception as e:
            st.error(f"Lỗi khi thực hiện suy luận: {e}")

# --------------------------------------------------------------------------- #
# TAB 2: QUÉT HÀNG LOẠT FILE CSV
# --------------------------------------------------------------------------- #
with tab2:
    st.subheader("Quét Rủi ro Hàng loạt từ File Giao dịch (Batch Scanning)")
    st.write("Tải lên file CSV chứa danh sách các giao dịch (cần có các cột `Time`, `V1`...`V28`, `Amount`).")

    uploaded_file = st.file_uploader("Chọn file CSV:", type=["csv"])

    col_use_sample, col_info = st.columns([1, 3])
    with col_use_sample:
        use_sample = st.button("📁 Dùng file mẫu 20 giao dịch")

    df_to_predict = None
    if uploaded_file is not None:
        df_to_predict = pd.read_csv(uploaded_file)
        st.success(f"Đã nạp file: {uploaded_file.name} ({len(df_to_predict)} dòng)")
    elif use_sample and sample_csv_path.exists():
        df_to_predict = pd.read_csv(sample_csv_path)
        st.info(f"Đã nạp file mẫu: sample_transactions.csv ({len(df_to_predict)} giao dịch)")

    if df_to_predict is not None:
        with st.spinner("Mô hình đang phân tích toàn bộ giao dịch..."):
            try:
                res_batch = predict(df_to_predict, threshold=threshold)
                n_total = len(res_batch)
                n_frauds = int(res_batch["is_fraud"].sum())
                fraud_pct = (n_frauds / n_total) * 100 if n_total > 0 else 0.0

                # Thống kê nhanh
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Tổng số giao dịch", f"{n_total:,}")
                c2.metric("Số giao dịch nghi gian lận", f"{n_frauds:,}")
                c3.metric("Tỷ lệ gắn cờ", f"{fraud_pct:.2f}%")
                c4.metric("Ngưỡng áp dụng", f"{threshold:.4f}")

                # Kiểm tra nếu file có cột nhãn thật Class
                if "Class" in res_batch.columns:
                    n_true_frauds = int(res_batch["Class"].sum())
                    tp = int(((res_batch["Class"] == 1) & (res_batch["is_fraud"] == 1)).sum())
                    recall_val = (tp / n_true_frauds) * 100 if n_true_frauds > 0 else 0.0
                    st.info(
                        f"📊 **Đối chiếu thực tế:** Tập dữ liệu có **{n_true_frauds}** ca gian lận thật. "
                        f"Mô hình đã bắt trúng **{tp}/{n_true_frauds}** ca (Tỷ lệ Recall đạt **{recall_val:.1f}%**)."
                    )

                # Hiển thị bảng kết quả
                st.markdown("#### Bảng chi tiết kết quả:")
                display_cols = ["fraud_probability", "is_fraud", "Amount", "Time"]
                if "Class" in res_batch.columns:
                    display_cols = ["Class"] + display_cols

                st.dataframe(
                    res_batch[display_cols].style.format({"fraud_probability": "{:.4f}", "Amount": "{:.2f}"}).apply(
                        lambda row: ["background-color: #ffcccc" if row.get("is_fraud", 0) == 1 else "" for _ in row],
                        axis=1,
                    ),
                    use_container_width=True,
                )

                # Nút download CSV kết quả
                csv_data = res_batch.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Tải xuống file kết quả phân tích (predictions.csv)",
                    data=csv_data,
                    file_name="predictions.csv",
                    mime="text/csv",
                )
            except Exception as ex:
                st.error(f"Lỗi khi xử lý file: {ex}")

# --------------------------------------------------------------------------- #
# TAB 3: BÁO CÁO ĐA MÔ HÌNH & HIỆU NĂNG
# --------------------------------------------------------------------------- #
with tab3:
    st.subheader("Bảng So sánh Định lượng Đa Mô hình (Rubric Tiêu chí 2 & 3)")
    st.markdown(
        """
        Nhóm 4 đã thử nghiệm **5 dòng thuật toán** kết hợp với 3 chiến lược mất cân bằng (`class_weight`, `SMOTE`, `Undersampling`).
        Toàn bộ được tinh chỉnh qua 5-Fold Stratified Cross-Validation với hàm mục tiêu tối ưu hóa **PR-AUC (Average Precision)**.
        """
    )

    reports_dir = Path(__file__).resolve().parent / "reports"
    cv_table_path = reports_dir / "tables" / "cv_comparison_tuned_models.csv"
    test_table_path = reports_dir / "tables" / "test_comparison_all_models.csv"

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("**Kết quả Cross-Validation (5-Fold trên tập Train):**")
        if cv_table_path.exists():
            df_cv = pd.read_csv(cv_table_path)
            st.dataframe(df_cv, use_container_width=True)
        else:
            st.info("Chưa tìm thấy bảng cv_comparison_tuned_models.csv")

    with col_t2:
        st.markdown("**Đánh giá Khách quan trên Tập Test Độc lập (56,746 giao dịch):**")
        if test_table_path.exists():
            df_test = pd.read_csv(test_table_path)
            st.dataframe(df_test, use_container_width=True)
        else:
            st.info("Chưa tìm thấy bảng test_comparison_all_models.csv")

    st.markdown("---")
    st.markdown("#### Các Đường cong Đánh giá Trực quan:")
    col_fig1, col_fig2 = st.columns(2)
    with col_fig1:
        fig_pr = reports_dir / "figures" / "16_test_pr_roc_curves.png"
        if fig_pr.exists():
            st.image(str(fig_pr), caption="Đường cong Precision-Recall & ROC trên tập Test", use_column_width=True)
    with col_fig2:
        fig_cm = reports_dir / "figures" / "17_confusion_matrices_test.png"
        if fig_cm.exists():
            st.image(str(fig_cm), caption="Ma trận Nhầm lẫn (Confusion Matrix) của các mô hình trên tập Test", use_column_width=True)

# --------------------------------------------------------------------------- #
# TAB 4: GIẢI THÍCH MÔ HÌNH (XAI & SHAP)
# --------------------------------------------------------------------------- #
with tab4:
    st.subheader("Khả năng Giải thích Mô hình Học máy (Explainable AI - XAI)")
    st.markdown(
        """
        Trong nghiệp vụ ngân hàng và kiểm soát rủi ro, một mô hình "hộp đen" là không đủ.
        Mô hình cần giải thích được **tại sao** một giao dịch bị coi là gian lận để tuân thủ pháp lý và hỗ trợ tra soát.
        """
    )

    col_shap1, col_shap2 = st.columns([3, 2])
    with col_shap1:
        fig_shap = reports_dir / "figures" / "20_shap_summary.png"
        if fig_shap.exists():
            st.image(str(fig_shap), caption="Biểu đồ SHAP Summary Plot (Độ đóng góp của từng đặc trưng)", use_column_width=True)
        else:
            fig_perm = reports_dir / "figures" / "19_permutation_importance.png"
            if fig_perm.exists():
                st.image(str(fig_perm), caption="Permutation Feature Importance", use_column_width=True)

    with col_shap2:
        st.markdown("#### 💡 Nhận xét Chuyên sâu từ SHAP:")
        st.markdown(
            """
            * **Các biến PCA phân biệt mạnh nhất:** $V_{14}$, $V_{10}$, $V_{12}$, $V_{17}$.
            * **Quy luật:** Khi giá trị của $V_{14}$ hoặc $V_{12}$ càng **âm sâu** (chấm màu xanh dương trên trục SHAP), giá trị SHAP càng **dương lớn** $\rightarrow$ Thúc đẩy mô hình dự đoán khả năng gian lận tăng vọt.
            * **Đặc trưng kỹ nghệ $V_{norm}$:** Chuẩn Euclidean tổng hợp $V_{norm} = \\sqrt{\\sum V_i^2}$ đứng trong top các đặc trưng quan trọng nhất, chứng minh giá trị của bước Kỹ nghệ đặc trưng mà nhóm đã thiết kế.
            * **Biến $Amount$ và $Amount\\_log$:** Không phải cứ giao dịch lớn là gian lận; tội phạm mạng thường chia nhỏ giao dịch hoặc quẹt số tiền rất nhỏ ($Amount < 1$) để dò thẻ.
            """
        )

