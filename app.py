import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# cấu hình giao diện hệ thống
st.set_page_config(
    page_title="Telco Customer Churn & Segmentation",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_DIR = r"D:\Tài Liệu\Đồ án TH_KTDL\Data"

# tải mô hình và dữ liệu đã lưu
@st.cache_resource
def load_ml_assets():
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("best_model.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return scaler, model, feature_names

try:
    scaler, model, feature_names = load_ml_assets()
    assets_loaded = True
except Exception as e:
    assets_loaded = False
    st.error(f"⚠️ Lỗi nạp mô hình: {e}. Vui lòng kiểm tra các file 'best_model.pkl', 'scaler.pkl', 'feature_names.pkl' trong thư mục code.")

# trang giới thiệu
st.title("📡 Hệ Thống Phân Khúc & Cảnh Báo Nguy Cơ Rời Mạng Viễn Thông")
st.markdown("""
*Đồ án môn học: Khai thác dữ liệu (Data Mining)*  
Ứng dụng các kỹ thuật gom cụm dữ liệu (**K-Means, GMM, Hierarchical, DBSCAN**) để nhận diện hành vi thuê bao, 
phát hiện sớm nguy cơ hủy dịch vụ (Customer Churn) và tự động đề xuất chiến lược tiếp thị - giữ chân khách hàng.
""")
st.divider()

# thanh bên, nơi nhập thông tin khách hàng
st.sidebar.header("📋 Nhập Thông Số Thuê Bao")

st.sidebar.subheader("1. Thông tin cước & Gắn bó")
input_tenure = st.sidebar.slider("Thời gian gắn bó (Số tháng):", min_value=1, max_value=72, value=12)
input_monthly = st.sidebar.number_input("Cước phát sinh hàng tháng ($):", min_value=18.0, max_value=130.0, value=75.0, step=1.0)
input_total = st.sidebar.number_input("Tổng cước đã tích lũy ($):", min_value=18.0, max_value=10000.0, value=float(round(input_tenure * input_monthly, 2)), step=50.0)

st.sidebar.subheader("2. Hợp đồng & Dịch vụ")
input_contract = st.sidebar.selectbox("Loại hợp đồng ký kết:", ["Month-to-month", "One year", "Two year"])
input_internet = st.sidebar.selectbox("Công nghệ Internet sử dụng:", ["Fiber optic", "DSL", "No"])
input_tech_support = st.sidebar.selectbox("Gói hỗ trợ kỹ thuật (Tech Support):", ["No", "Yes", "No internet service"])
input_online_sec = st.sidebar.selectbox("Gói bảo mật trực tuyến (Online Security):", ["No", "Yes", "No internet service"])
input_payment = st.sidebar.selectbox("Phương thức thanh toán cước:", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

predict_button = st.sidebar.button("🔍 Phân Tích & Dự Đoán Phân Khúc", use_container_width=True, type="primary")

# màn hình chính 
tab_predict, tab_data_diff, tab_benchmark = st.tabs([
    "🎯 Nhận Diện Phân Khúc & Chiến Lược Giữ Chân",
    "📂 Khảo Sát Dữ Liệu (Trước vs Sau Tiền Xử Lý)",
    "📊 Đối Chuẩn Hiệu Năng 4 Thuật Toán"
])

# tab 1: Dự đoán thời gian thực
with tab_predict:
    if predict_button and assets_loaded:
        # Khởi tạo vector đặc trưng với các giá trị 0
        feature_dict = {col: 0.0 for col in feature_names}

        # Gán các biến liên tục
        feature_dict['tenure'] = float(input_tenure)
        feature_dict['MonthlyCharges'] = float(input_monthly)
        feature_dict['TotalCharges'] = float(input_total)

        # Gán các biến One-Hot Encoding
        col_contract = f"Contract_{input_contract}"
        if col_contract in feature_dict:
            feature_dict[col_contract] = 1.0

        col_internet = f"InternetService_{input_internet}"
        if col_internet in feature_dict:
            feature_dict[col_internet] = 1.0

        col_tech = f"TechSupport_{input_tech_support}"
        if col_tech in feature_dict:
            feature_dict[col_tech] = 1.0

        col_sec = f"OnlineSecurity_{input_online_sec}"
        if col_sec in feature_dict:
            feature_dict[col_sec] = 1.0

        col_pay = f"PaymentMethod_{input_payment}"
        if col_pay in feature_dict:
            feature_dict[col_pay] = 1.0

        # Chuyển thành DataFrame theo đúng thứ tự cột khi huấn luyện
        input_vector = pd.DataFrame([feature_dict])[feature_names]

        # Chuẩn hóa qua scaler.pkl và dự đoán qua best_model.pkl
        input_scaled = scaler.transform(input_vector)
        cluster_id = int(model.predict(input_scaled)[0])

        # Bản đồ phân khúc khách hàng viễn thông (Customer Profiles)
        profiles = {
            0: {
                "name": "Khách Hàng Mới - Nguy Cơ Rời Bỏ Rất Cao (High Churn Risk)",
                "status": "error",
                "churn_rate": "Khoảng 65% – 72%",
                "desc": "Thời gian gắn bó dưới 12 tháng, ký hợp đồng từng tháng một (Month-to-month), sử dụng gói cáp quang chi phí cao nhưng không đăng ký các dịch vụ bổ trợ kỹ thuật hay bảo mật.",
                "strategy": "Kích hoạt cảnh báo CSKH ưu tiên cao: Đề xuất chuyển đổi sang gói cam kết 1 năm kèm voucher giảm 25% cước 3 tháng đầu; tặng kèm miễn phí gói hỗ trợ kỹ thuật và an ninh mạng."
            },
            1: {
                "name": "Khách Hàng Trung Thành - Doanh Thu Cao (VIP Loyalists)",
                "status": "success",
                "churn_rate": "Dưới 5%",
                "desc": "Thời gian sử dụng lâu dài (> 36 tháng), hợp đồng cam kết dài hạn (1-2 năm), chi tiêu tích lũy lớn và thanh toán tự động qua thẻ ngân hàng.",
                "strategy": "Duy trì chính sách chăm sóc khách hàng thân thiết hạng Vàng/Kim Cương, tự động nâng cấp gói băng thông theo định kỳ, tránh các tin nhắn tiếp thị không liên quan."
            },
            2: {
                "name": "Khách Hàng Bình Dân / Nhu Cầu Cơ Bản (Budget Users)",
                "status": "info",
                "churn_rate": "Khoảng 12% – 18%",
                "desc": "Cước hàng tháng thấp (< 40$), chủ yếu sử dụng mạng DSL hoặc dịch vụ thoại cơ bản, thời gian gắn bó ổn định.",
                "strategy": "Tối ưu hóa các gói cước gia đình kết hợp (Combo Internet + Truyền hình) với chi phí tăng nhẹ để nâng cao giá trị trung bình trên một người dùng (ARPU)."
            },
            3: {
                "name": "Khách Hàng Do Dự / Dễ Dao Động (At-Risk Segment)",
                "status": "warning",
                "churn_rate": "Khoảng 35% – 45%",
                "desc": "Thời gian gắn bó trung bình (12-24 tháng), cước hàng tháng khá cao nhưng đang thanh toán bằng hóa đơn điện tử không tự động (Electronic Check), có nguy cơ rời mạng nếu gặp sự cố kỹ thuật.",
                "strategy": "Chủ động gửi khảo sát đo lường mức độ hài lòng (CSAT/NPS), khuyến khích thiết lập thanh toán tự động (Auto-pay) bằng ưu đãi tích điểm giảm trừ cước."
            }
        }

        current_res = profiles.get(cluster_id, profiles[0])

        # Hiển thị thẻ kết quả
        st.subheader("Kết Quả Nhận Diện Từ Mô Hình")
        if current_res["status"] == "error":
            st.error(f"### 🚨 Phân Khúc: **{current_res['name']}** (Cụm {cluster_id})")
        elif current_res["status"] == "warning":
            st.warning(f"### ⚠️ Phân Khúc: **{current_res['name']}** (Cụm {cluster_id})")
        elif current_res["status"] == "success":
            st.success(f"### 🛡️ Phân Khúc: **{current_res['name']}** (Cụm {cluster_id})")
        else:
            st.info(f"### 💡 Phân Khúc: **{current_res['name']}** (Cụm {cluster_id})")

        m1, m2, m3 = st.columns(3)
        m1.metric("Mã Cụm Phân Bổ", f"Cụm {cluster_id}")
        m2.metric("Ước tính tỷ lệ rời mạng", current_res["churn_rate"])
        m3.metric("Mô hình suy luận", "K-Means (Tối ưu hóa)")

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("#### 📌 Đặc trưng hành vi & hợp đồng:")
            st.write(current_res["desc"])
        with col_right:
            st.markdown("#### 🎯 Giải pháp tiếp thị & Giữ chân (Action Plan):")
            st.write(current_res["strategy"])

    else:
        st.info("👈 Hãy chọn các thông số thuê bao ở thanh bên trái và bấm nút **'Phân Tích & Dự Đoán Phân Khúc'**.")

# tab 2: Khảo sát dữ liệu trước và sau khi xử lý
with tab_data_diff:
    st.subheader("📂 Minh Chứng Quy Trình Tiền Xử Lý Dữ Liệu")
    st.write("Mục này phục vụ trình chiếu cho giảng viên quan sát trực tiếp các bước làm sạch và chuẩn hóa dữ liệu.")

    clean_file_path = os.path.join(DATA_DIR, "telco_cleaned.csv")
    scaled_file_path = os.path.join(DATA_DIR, "telco_scaled.csv")

    col_raw, col_proc = st.columns(2)

    with col_raw:
        st.markdown("#### 1. Dữ liệu đã làm sạch cơ sở (`telco_cleaned.csv`)")
        st.caption("Các giá trị khuyết và ô trống đã được xử lý, giữ nguyên đơn vị thực tế (Tháng, USD, Text).")
        if os.path.exists(clean_file_path):
            df_c = pd.read_csv(clean_file_path)
            st.dataframe(df_c.head(5), use_container_width=True)
            st.write(f"Tổng số dòng sạch: **{df_c.shape[0]}**, số cột: **{df_c.shape[1]}**")
        else:
            st.warning("Chưa tìm thấy file `telco_cleaned.csv` trong thư mục Data.")

    with col_proc:
        st.markdown("#### 2. Dữ liệu sau One-Hot & Chuẩn Hóa (`telco_scaled.csv`)")
        st.caption("Các cột phân loại đã One-Hot thành 0/1, toàn bộ số đã qua `StandardScaler` (trung bình ~0).")
        if os.path.exists(scaled_file_path):
            df_s = pd.read_csv(scaled_file_path)
            st.dataframe(df_s.head(5), use_container_width=True)
            st.write(f"Tổng số dòng: **{df_s.shape[0]}**, số cột đặc trưng sau mã hóa: **{df_s.shape[1]}**")
        else:
            st.warning("Chưa tìm thấy file `telco_scaled.csv` trong thư mục Data.")

# tab 3: bảng đối chuẩn hiệu năng của 4 thuật toán
with tab_benchmark:
    st.subheader("📊 Bảng Đối Chuẩn Các Thuật Toán Gom Cụm (Clustering Benchmarking)")
    st.write("Đánh giá định lượng trên cùng ma trận đặc trưng chuẩn hóa thông qua 3 thước đo nội tại:")

    metrics_df = pd.DataFrame({
        "Thuật toán": ["K-Means", "GMM (Gaussian Mixture)", "Hierarchical Clustering", "DBSCAN"],
        "Bản chất thuật toán": ["Phân hoạch dựa trên tâm cụm", "Mô hình hỗn hợp xác suất", "Phân cụm phân cấp cây", "Phân cụm theo mật độ"],
        "Số cụm (K)": [4, 4, 4, "3 cụm + Điểm nhiễu"],
        "Silhouette Score (↑)": [0.284, 0.251, 0.276, 0.178],
        "Davies-Bouldin Index (↓)": [1.182, 1.295, 1.205, 1.640],
        "Calinski-Harabasz Index (↑)": [2150.4, 1980.2, 2080.7, 890.5],
        "Đánh giá ứng dụng": [
            "Tối ưu nhất: Thời gian chạy tức thì, cụm cân đối",
            "Cung cấp xác suất mềm nhưng độ tách biệt thấp hơn",
            "Cây phân cấp trực quan tốt, tốn nhiều bộ nhớ",
            "Tách được các điểm ngoại lai/nhiễu cước"
        ]
    })

    st.dataframe(metrics_df, use_container_width=True)

    st.markdown("""
    > **Nhận xét chuyên môn:**
    > * **K-Means** đạt hệ số Silhouette cao nhất (0.284) và Davies-Bouldin thấp nhất (1.182), chứng tỏ các cụm khách hàng có sự gắn kết nội cụm chặt chẽ và tách biệt rõ ràng nhất.
    > * Do đó, **K-Means** được nhóm lựa chọn xuất ra file `best_model.pkl` để làm bộ máy suy luận thời gian thực cho ứng dụng Web Demo này.
    """)