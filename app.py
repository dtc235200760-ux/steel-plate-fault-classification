
import os
import html
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import base64

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# CÁC HÀM HỖ TRỢ GIAO DIỆN
# ============================================================

def page_head(title, subtitle=""):
    st.markdown(
        f"""
        <div class="main-title">
            <h1>{html.escape(str(title))}</h1>
            <p>{html.escape(str(subtitle))}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def hero(title, description):
    st.markdown(
        f"""
        <div class="info-card">
            <h3>{html.escape(str(title))}</h3>
            <p>{html.escape(str(description))}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# HÀM HỖ TRỢ DỰ ĐOÁN HÀNG LOẠT
# =========================================================

def default_model_index():
    """Ưu tiên chọn Random Forest."""
    model_names = list(models.keys())

    if "Random Forest" in model_names:
        return model_names.index("Random Forest")

    return 0


def read_uploaded_csv(uploaded_file):
    """Đọc file CSV người dùng tải lên."""
    try:
        df_upload = pd.read_csv(
            uploaded_file,
            encoding="utf-8-sig"
        )
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        df_upload = pd.read_csv(
            uploaded_file,
            encoding="latin1"
        )

    df_upload.columns = (
        df_upload.columns.astype(str).str.strip()
    )

    if df_upload.empty:
        raise ValueError("File CSV không có dữ liệu.")

    return df_upload


def prepare_input_data(input_df, feature_names):
    """Kiểm tra và chuẩn hóa dữ liệu đầu vào."""

    missing = [
        col for col in feature_names
        if col not in input_df.columns
    ]

    if missing:
        raise ValueError(
            f"File CSV thiếu các cột đặc trưng: {missing}"
        )

    X_input = input_df[feature_names].copy()

    for col in feature_names:
        X_input[col] = pd.to_numeric(
            X_input[col],
            errors="coerce"
        )

    X_input = X_input.replace(
        [np.inf, -np.inf],
        np.nan
    )

    if X_input.isna().any().any():
        invalid_cols = X_input.columns[
            X_input.isna().any()
        ].tolist()

        raise ValueError(
            f"Dữ liệu có ô trống hoặc không phải số "
            f"ở các cột: {invalid_cols}"
        )

    return X_input


# Bảng màu dùng cho biểu đồ
PALETTE = [
    "#1677ff",
    "#13b8a6",
    "#f59e0b",
    "#ef4444",
    "#8b5cf6",
    "#ec4899",
    "#64748b"
]


def style_fig(fig, height=400):
    """Định dạng biểu đồ Plotly."""
    fig.update_layout(
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(
            family="Arial",
            color="#172b4d"
        ),
        margin=dict(
            l=20,
            r=30,
            t=50,
            b=30
        )
    )

    return fig

# =========================================================
# 1. CẤU HÌNH
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOGO_PATH = os.path.join(BASE_DIR, "logo.png")

# Đọc và mã hóa ảnh logo
with open(LOGO_PATH, "rb") as f:
    logo_base64 = base64.b64encode(f.read()).decode("utf-8")

st.set_page_config(
    page_title="Phân loại lỗi bề mặt thép",
    page_icon=LOGO_PATH,
    layout="wide"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")
# Chỉ sử dụng file dữ liệu này cho chức năng chọn mẫu
DATA_FILE = os.path.join(
    RESULT_DIR,
    "du_lieu_da_xu_ly.csv"
)

MODEL_FILES = {
    "SVM": "svm.joblib",
    "KNN": "knn.joblib",
    "Random Forest": "random_forest.joblib",
    "Extra Trees": "extra_trees.joblib",
    "Gradient Boosting": "gradient_boosting.joblib"
}

CLASS_NAMES = {
    "Bumps": "Lỗi dạng lồi, gồ ghề",
    "Dirtiness": "Lỗi bề mặt bẩn",
    "K_Scratch": "Lỗi vết xước dạng K",
    "Other_Faults": "Các lỗi khác",
    "Pastry": "Lỗi dạng vết khuyết không đều",
    "Stains": "Lỗi vết ố",
    "Z_Scratch": "Lỗi vết xước dạng Z"
}

FEATURE_GROUPS = {
    "Vị trí": [
        "X_Minimum",
        "X_Maximum",
        "Y_Minimum",
        "Y_Maximum"
    ],

    "Diện tích và chu vi": [
        "Pixels_Areas",
        "X_Perimeter",
        "Y_Perimeter"
    ],

    "Độ sáng": [
        "Sum_of_Luminosity",
        "Minimum_of_Luminosity",
        "Maximum_of_Luminosity",
        "Luminosity_Index"
    ],

    "Hình dạng và đường biên": [
        "Edges_Index",
        "Empty_Index",
        "Square_Index",
        "Outside_X_Index",
        "Edges_X_Index",
        "Edges_Y_Index",
        "Outside_Global_Index",
        "Orientation_Index"
    ],

    "Biến đổi": [
        "LogOfAreas",
        "Log_X_Index",
        "Log_Y_Index",
        "SigmoidOfAreas"
    ],

    "Vật liệu và sản xuất": [
        "Length_of_Conveyer",
        "TypeOfSteel_A300",
        "TypeOfSteel_A400",
        "Steel_Plate_Thickness"
    ]
}

# =========================================================
# 2. GIAO DIỆN CSS
# =========================================================
st.markdown("""
<style>
/* Nền tổng thể */
.stApp {
    background-color: #f3f7fc;
    color: #1e293b;
    font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}

[data-testid="stDecoration"] {
    display: none !important;
}

/* Thanh điều hướng bên trái (Sidebar) */
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e2e8f0;
}

.sidebar-brand {
    text-align: center;
    padding: 15px 10px 20px 10px;
    border-bottom: 1px solid #f1f5f9;
    margin-bottom: 15px;
}

.sidebar-brand-icon {
    font-size: 45px;
    color: #1677ff;
    margin-bottom: 5px;
}

.sidebar-brand-title {
    font-size: 24px;
    font-weight: 800;
    color: #0f172a;
    margin: 0;
}

.sidebar-brand-sub {
    font-size: 16px;
    color: #64748b;
    margin-top: 2px;
}

/* Style lại nút Radio trên Sidebar thành dạng menu bo tròn */
[data-testid="stSidebar"] div[role="radiogroup"] > label {
    padding: 10px 14px;
    border-radius: 10px;
    margin-bottom: 6px;
    transition: all 0.2s;
}

/* Ép cỡ chữ to cho các mục menu bên trong Sidebar */
[data-testid="stSidebar"] div[role="radiogroup"] label p,
[data-testid="stSidebar"] div[role="radiogroup"] label span,
[data-testid="stSidebar"] div[role="radiogroup"] label div {
    font-size: 18px !important;
    font-weight: 600 !important;
    color: #1e293b !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background-color: #eff6ff;
}

/* Khung Header chính (Chỉ chứa 1 ảnh bên trái) */
.steel-header {
    background: linear-gradient(120deg, #e0f2fe 0%, #f0f9ff 60%, #e0e7ff 100%);
    padding: 22px 28px;
    border-radius: 16px;
    border: 1px solid #bae6fd;
    margin-bottom: 22px;
    box-shadow: 0 4px 12px rgba(14, 165, 233, 0.06);
}

.steel-header-content {
    display: flex;
    align-items: center;
    gap: 22px;
}

.steel-logo {
    width: 250px;        
    height: 200px;       
    object-fit: cover;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.12);
    border: 2px solid #ffffff;
    flex-shrink: 0;
}

.steel-title {
    color: #0f172a;
    font-size: 28px !important;
    font-weight: 800;
    margin: 0;
    letter-spacing: 0.3px;
}

.steel-tagline {
    color: #0284c7;
    font-size: 20px !important;
    font-weight: 600;
    font-style: italic;
    margin: 4px 0 2px 0;
}

/* Tăng cỡ chữ phụ đề dưới tiêu đề chính */
.steel-subtitle {
    color: #334155 !important;
    font-size: 18px !important;
    font-weight: 500 !important;
    margin-top: 6px !important;
}

/* 4 Thẻ chỉ số (KPI Cards) */
.kpi-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 18px 20px;
    display: flex;
    align-items: center;
    gap: 16px;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    border: 1px solid #e2e8f0;
    margin-bottom: 18px;
}

.kpi-icon {
    width: 56px;
    height: 56px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 26px;
    flex-shrink: 0;
}

.kpi-blue { background: #e0f2fe; color: #0284c7; }
.kpi-teal { background: #ccfbf1; color: #0d9488; }
.kpi-amber { background: #fef3c7; color: #d97706; }
.kpi-purple { background: #f3e8ff; color: #7e22ce; }

/* Tăng cỡ chữ nhãn của 4 thẻ chỉ số */
.kpi-label {
    font-size: 17px !important;
    color: #475569 !important;
    font-weight: 700 !important;
    margin-bottom: 4px !important;
}

.kpi-value {
    font-size: 28px !important;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.1;
}

/* Khung nội dung trắng (Card Container) */
.dashboard-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 20px 24px;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
    border: 1px solid #e2e8f0;
    margin-bottom: 20px;
    height: 100%;
}

.card-title {
    font-size: 18px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.card-sub {
    font-size: 14px;
    color: #64748b;
    margin-bottom: 14px;
}

/* Danh sách check thông tin dữ liệu */
.info-list {
    list-style: none;
    padding: 0;
    margin: 12px 0 0 0;
}

.info-list li {
    font-size: 15px;
    color: #334155;
    margin-bottom: 10px;
    display: flex;
    align-items: flex-start;
    gap: 8px;
}

/* Lưới 5 mô hình học máy */
.ml-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 10px;
    margin-top: 14px;
}

.ml-badge {
    border-radius: 12px;
    padding: 14px 6px;
    text-align: center;
    font-weight: 700;
    font-size: 13px;
    color: #1e293b;
}

.ml-icon {
    font-size: 22px;
    display: block;
    margin-bottom: 6px;
}

/* Các thẻ thông báo cho trang con */
.main-title {
    background: linear-gradient(120deg, #0284c7, #0d9488);
    padding: 20px 24px;
    border-radius: 14px;
    color: white;
    margin-bottom: 20px;
}
.main-title h1 { color: white; margin: 0; font-size: 22px; }
.main-title p { color: #e0f2fe; margin: 4px 0 0 0; font-size: 15px; }

.info-card {
    background: white;
    padding: 18px;
    border-radius: 13px;
    border: 1px solid #e2e8f0;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 3. TẢI MÔ HÌNH
# =========================================================

@st.cache_resource
def load_models():

    models = {}
    errors = []

    for name, filename in MODEL_FILES.items():

        path = os.path.join(MODEL_DIR, filename)

        if os.path.exists(path):

            try:
                models[name] = joblib.load(path)

            except Exception as e:
                errors.append(f"{name}: {e}")

        else:
            errors.append(f"Không tìm thấy {filename}")

    return models, errors


@st.cache_resource
def load_feature_names():

    path = os.path.join(
        MODEL_DIR,
        "feature_names.joblib"
    )

    if os.path.exists(path):
        return joblib.load(path)

    return None


@st.cache_data
def load_processed_data():

    if not os.path.exists(DATA_FILE):
        return None, f"Không tìm thấy file: {DATA_FILE}"

    try:

        df = pd.read_csv(
            DATA_FILE,
            encoding="utf-8-sig"
        )

        df.columns = df.columns.astype(str).str.strip()

        if "Loai_loi" not in df.columns:
            return None, "File dữ liệu thiếu cột Loai_loi."

        return df, None

    except Exception as e:
        return None, str(e)


models, model_errors = load_models()
feature_names = load_feature_names()
df, data_error = load_processed_data()


# =========================================================
# 4. KIỂM TRA DỮ LIỆU
# =========================================================

if feature_names is None:

    st.error(
        "Không tìm thấy models/feature_names.joblib."
    )

    st.stop()


feature_names = list(feature_names)

if df is None:

    st.error(data_error)

    st.info(
        "Hãy đặt file du_lieu_da_xu_ly.csv "
        "vào thư mục results."
    )

    st.stop()


missing_features = [
    col for col in feature_names
    if col not in df.columns
]

if missing_features:

    st.error("File dữ liệu đang thiếu các cột đặc trưng:")

    st.write(missing_features)

    st.stop()


if not models:

    st.error(
        "Không tải được mô hình nào. "
        "Hãy kiểm tra thư mục models."
    )

    st.write(model_errors)

    st.stop()


# Chuyển dữ liệu đặc trưng về dạng số

X_all = df[feature_names].apply(
    pd.to_numeric,
    errors="coerce"
)

X_all = X_all.replace(
    [np.inf, -np.inf],
    np.nan
)

valid_mask = (
    X_all.notna().all(axis=1)
    & df["Loai_loi"].notna()
)

X_all = X_all.loc[valid_mask].reset_index(drop=True)

y_all = (
    df.loc[valid_mask, "Loai_loi"]
    .astype(str)
    .reset_index(drop=True)
)


if len(X_all) == 0:

    st.error("Không có mẫu dữ liệu hợp lệ.")

    st.stop()


# =========================================================
# 5. HÀM HỖ TRỢ
# =========================================================

def translate_label(label):

    return CLASS_NAMES.get(
        str(label),
        str(label)
    )


def predict_all_models(sample):

    results = {}

    sample_df = pd.DataFrame(
        [sample],
        columns=feature_names
    )

    for model_name, model in models.items():

        try:

            prediction = model.predict(sample_df)[0]

            results[model_name] = str(prediction)

        except Exception as e:

            results[model_name] = f"Lỗi: {e}"

    return results


def calculate_model_results():

    rows = []

    for model_name, model in models.items():

        try:

            predictions = model.predict(X_all)

            rows.append({

                "Mô hình": model_name,

                "Accuracy": accuracy_score(
                    y_all,
                    predictions
                ),

                "Precision Macro": precision_score(
                    y_all,
                    predictions,
                    average="macro",
                    zero_division=0
                ),

                "Recall Macro": recall_score(
                    y_all,
                    predictions,
                    average="macro",
                    zero_division=0
                ),

                "F1 Macro": f1_score(
                    y_all,
                    predictions,
                    average="macro",
                    zero_division=0
                )

            })

        except Exception:
            continue

    return pd.DataFrame(rows)


def get_predictions_by_class():

    rows = []

    for model_name, model in models.items():

        try:

            predictions = model.predict(X_all)

            for label in sorted(y_all.unique()):

                mask = (
                    y_all.to_numpy() == label
                )

                total = int(mask.sum())

                correct = int(
                    (
                        predictions[mask]
                        == y_all.to_numpy()[mask]
                    ).sum()
                )

                rows.append({

                    "Mô hình": model_name,

                    "Loại lỗi": label,

                    "Tên lỗi": translate_label(label),

                    "Tổng số mẫu": total,

                    "Dự đoán đúng": correct,

                    "Tỷ lệ đúng (%)": round(
                        correct / total * 100,
                        2
                    ) if total else 0

                })

        except Exception:
            continue

    return pd.DataFrame(rows)


# =========================================================
# 6. THANH ĐIỀU HƯỚNG
# =========================================================
st.sidebar.markdown("""
<div class="sidebar-brand">
<div class="sidebar-brand-title">Hệ thống phân loại lỗi bề mặt tấm thép</div>
<div class="sidebar-brand-sub">Thông minh hơn cho ngành thép</div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "DANH MỤC",
    [
        "Tổng quan",
        "Dự đoán lỗi",
        "Dự đoán hàng loạt",
        "So sánh mô hình",
        "Thông tin hệ thống"
    ],
    label_visibility="collapsed"
)

st.sidebar.divider()
st.sidebar.success(f"Đã tải {len(models)}/5 mô hình")
st.sidebar.caption(f"Số mẫu dữ liệu: {len(X_all):,}")
# =========================================================
# 7. TIÊU ĐỀ CHUNG
# =========================================================
st.markdown(f"""
<div class="steel-header">
<div class="steel-header-content">
<img class="steel-logo" src="data:image/png;base64,{logo_base64}">
<div>
<h1 class="steel-title">PHÂN LOẠI LỖI BỀ MẶT TẤM THÉP</h1>
<div class="steel-subtitle">Ứng dụng học máy sử dụng đặc trưng hình học và độ sáng</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# 8. TRANG TỔNG QUAN
# =========================================================

if page == "Tổng quan":

    #  4 thẻ chỉ số KPI ---
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="kpi-card">
        <div class="kpi-icon kpi-blue">🗄️</div>
        <div>
        <div class="kpi-label">Tổng số mẫu</div>
        <div class="kpi-value">{len(X_all):,}</div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="kpi-card">
        <div class="kpi-icon kpi-teal">📊</div>
        <div>
        <div class="kpi-label">Số đặc trưng</div>
        <div class="kpi-value">{len(feature_names)}</div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="kpi-card">
        <div class="kpi-icon kpi-amber">📑</div>
        <div>
        <div class="kpi-label">Loại lỗi</div>
        <div class="kpi-value">{y_all.nunique()}</div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="kpi-card">
        <div class="kpi-icon kpi-purple">🧠</div>
        <div>
        <div class="kpi-label">Mô hình ML</div>
        <div class="kpi-value">{len(models)}</div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    class_counts = (
        y_all.value_counts()
        .rename_axis("Loại lỗi")
        .reset_index(name="Số lượng")
    )

    chart_colors = ["#2563eb", "#14b8a6", "#f59e0b", "#a855f7", "#ef4444", "#22c55e", "#64748b"]

    fig = px.bar(
        class_counts,
        x="Loại lỗi",
        y="Số lượng",
        color="Loại lỗi",
        text="Số lượng",
        color_discrete_sequence=chart_colors
    )

    fig.update_traces(
        textposition="outside",
        textfont=dict(size=13, color="#1e293b", family="Arial"),
        marker_line_width=0
    )

    fig.update_layout(
        height=320,
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        xaxis_title="",
        yaxis_title="Số lượng mẫu",
        margin=dict(l=10, r=10, t=25, b=10),
        yaxis=dict(gridcolor="#f1f5f9", range=[0, class_counts["Số lượng"].max() * 1.15])
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    #  2 cột Thông tin dữ liệu & 5 mô hình học máy ---
    col_left, col_right = st.columns([1.05, 1])

    with col_left:
        st.markdown(f"""
        <div class="dashboard-card">
        <div class="card-title">📈 Thông tin dữ liệu</div>
        <ul class="info-list">
        <li>✅ <span><b>Tập dữ liệu:</b> Steel Plates Faults (UCI)</span></li>
        <li>✅ <span><b>Quy mô:</b> {len(X_all):,} mẫu dữ liệu, {len(feature_names)} đặc trưng hình học & độ sáng</span></li>
        <li>✅ <span><b>7 loại lỗi:</b> Other_Faults, Bumps, K_Scratch, Z_Scratch, Pastry, Stains, Dirtiness</span></li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class="dashboard-card">
        <div class="card-title">🏆 5 mô hình học máy</div>
        <div class="card-sub">Các thuật toán phân loại được huấn luyện và đối chiếu</div>
        <div class="ml-grid">
            <div class="ml-badge" style="background:#e0f2fe; color:#0369a1;">
                <span class="ml-icon">💠</span>SVM
            </div>
            <div class="ml-badge" style="background:#ccfbf1; color:#0f766e;">
                <span class="ml-icon">🕸️</span>KNN
            </div>
            <div class="ml-badge" style="background:#fef3c7; color:#b45309;">
                <span class="ml-icon">🌲</span>Random Forest
            </div>
            <div class="ml-badge" style="background:#f3e8ff; color:#6b21a8;">
                <span class="ml-icon">🌳</span>Extra Trees
            </div>
            <div class="ml-badge" style="background:#dcfce7; color:#15803d;">
                <span class="ml-icon">🚀</span>Gradient Boosting
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    # --- Bảng xem trước dữ liệu (đặt trong expander cho gọn giao diện) ---
    with st.expander("📋 Xem bảng dữ liệu mẫu (10 dòng đầu tiên)"):
        preview = X_all.copy()
        preview["Loai_loi"] = y_all
        st.dataframe(preview.head(10), use_container_width=True)
# =========================================================
# 9. TRANG DỰ ĐOÁN LỖI
# =========================================================

elif page == "Dự đoán lỗi":

    st.subheader("🔍 Dự đoán và so sánh mô hình")

    input_mode = st.radio(
        "Chọn cách nhập dữ liệu",
        [
            "Chọn mẫu từ dữ liệu có sẵn",
            "Nhập thông số thủ công"
        ],
        horizontal=True
    )

    sample = None
    actual_label = None
    sample_description = ""

    # -----------------------------------------------------
    # Chọn mẫu có sẵn
    # -----------------------------------------------------

    if input_mode == "Chọn mẫu từ dữ liệu có sẵn":
        st.markdown("### Chọn loại lỗi")

        available_classes = sorted(
            y_all.unique()
        )

        selected_class = st.selectbox(
            "Tên loại lỗi",
            available_classes,
            format_func=lambda x: (
                f"{x} - {translate_label(x)}"
            )
        )

        class_indices = y_all[
            y_all == selected_class
        ].index.tolist()

        selected_index = st.selectbox(
            "Chọn mẫu dữ liệu",
            class_indices,
            format_func=lambda i: (
                f"Mẫu {i + 1} - {selected_class}"
            )
        )

        sample = X_all.iloc[
            selected_index
        ].to_dict()

        actual_label = y_all.iloc[
            selected_index
        ]

        sample_description = (
            f"Mẫu {selected_index + 1}"
        )

        st.markdown("### Thông tin mẫu đã chọn")

        st.write(
            f"**Nhãn lỗi thực tế:** "
            f"{actual_label} - "
            f"{translate_label(actual_label)}"
        )

        with st.expander(
            "Xem 27 đặc trưng của mẫu"
        ):

            st.dataframe(
                pd.DataFrame([sample]),
                width="stretch"
            )

    # -----------------------------------------------------
    # Nhập thủ công
    # -----------------------------------------------------

    else:

        st.markdown("### Nhập thông số đặc trưng")

        sample = {}

        for group_name, columns in FEATURE_GROUPS.items():

            valid_columns = [
                col for col in columns
                if col in feature_names
            ]

            if not valid_columns:
                continue

            with st.expander(
                group_name,
                expanded=True
            ):

                cols = st.columns(2)

                for i, col in enumerate(valid_columns):

                    with cols[i % 2]:

                        sample[col] = st.number_input(
                            col,
                            value=0.0,
                            format="%.6f",
                            key=f"manual_{col}"
                        )

        actual_label = None

        sample_description = "Mẫu nhập thủ công"

    st.divider()

    # -----------------------------------------------------
    # Dự đoán
    # -----------------------------------------------------

    if st.button(
        "🚀 DỰ ĐOÁN VÀ SO SÁNH 5 MÔ HÌNH",
        type="primary",
        width="stretch"
    ):

        if sample is None:

            st.error(
                "Không có dữ liệu để dự đoán."
            )

        else:

            results = predict_all_models(sample)

            st.markdown(
                "### Kết quả dự đoán của từng mô hình"
            )

            result_rows = []

            for model_name, prediction in results.items():

                is_correct = (
                    actual_label is not None
                    and prediction == actual_label
                )

                result_rows.append({

                    "Mô hình": model_name,

                    "Kết quả dự đoán": prediction,

                    "Tên lỗi dự đoán": translate_label(
                        prediction
                    ),

                    "So với nhãn thực tế": (
                        "Đúng"
                        if is_correct
                        else "Sai"
                        if actual_label is not None
                        else "Chưa có nhãn đối chiếu"
                    )

                })

            result_df = pd.DataFrame(
                result_rows
            )

            st.dataframe(
                result_df,
                width="stretch",
                hide_index=True
            )

            if actual_label is not None:

                correct_count = sum(
                    prediction == actual_label
                    for prediction in results.values()
                )

                                # Hiển thị thông tin đối chiếu với nhãn thực tế
                with st.container(border=True):

                    st.subheader("✅ Đối chiếu với nhãn thực tế")

                    st.write(
                        f"**Mẫu:** {sample_description}"
                    )

                    st.write(
                        f"**Nhãn thực tế:** {actual_label} - "
                        f"{translate_label(actual_label)}"
                    )

                    st.write(
                        f"**Số mô hình dự đoán đúng:** "
                        f"{correct_count}/{len(results)}"
                    )

                if correct_count == len(results):

                    st.success(
                        "Cả 5 mô hình đều dự đoán đúng mẫu này."
                    )

                elif correct_count == 0:

                    st.warning(
                        "Cả 5 mô hình đều dự đoán sai mẫu này."
                    )

                else:

                    st.info(
                        f"Có {correct_count} mô hình dự đoán đúng "
                        f"và {len(results) - correct_count} "
                        f"mô hình dự đoán sai."
                    )

            else:

                st.info(
                    "Đây là mẫu nhập thủ công nên không có "
                    "nhãn thực tế để xác định mô hình nào "
                    "dự đoán đúng."
                )

            st.markdown(
                "### Biểu đồ kết quả dự đoán"
            )

            chart_df = pd.DataFrame([

                {
                    "Mô hình": name,

                    "Loại lỗi dự đoán": translate_label(
                        prediction
                    )
                }

                for name, prediction in results.items()

            ])

            fig = px.bar(
                chart_df,
                x="Mô hình",
                y=[1] * len(chart_df),
                color="Loại lỗi dự đoán",
                title="Kết quả phân loại của các mô hình"
            )

            fig.update_layout(
                yaxis_title="Kết quả",
                yaxis_visible=False,
                plot_bgcolor="white",
                paper_bgcolor="white"
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

# =========================================================
# 10. TRANG DỰ ĐOÁN HÀNG LOẠT
# =========================================================

elif page == "Dự đoán hàng loạt":

    page_head(
        "Dự đoán lỗi từ file CSV",
        "Tải lên dữ liệu CSV chứa các đặc trưng của nhiều tấm thép."
    )

    # -----------------------------------------------------
    # CHỌN MÔ HÌNH
    # -----------------------------------------------------

    left, right = st.columns([2, 1])

    with left:
        selected_model_name = st.selectbox(
            "Mô hình dự đoán",
            list(models.keys()),
            index=default_model_index(),
            key="batch_model"
        )

    selected_model = models[selected_model_name]

    with right:
        st.write("")

        st.download_button(
            "⬇ Tải file CSV mẫu",
            data=pd.DataFrame(
                columns=feature_names
            ).to_csv(
                index=False,
                encoding="utf-8-sig"
            ),
            file_name="mau_dac_trung_thep.csv",
            mime="text/csv",
            width="stretch"
        )

    # -----------------------------------------------------
    # TẢI FILE CSV
    # -----------------------------------------------------

    uploaded_file = st.file_uploader(
        "Chọn file CSV",
        type=["csv"],
        key="batch_upload"
    )

    if uploaded_file is None:

        st.info(
            "Chưa có file. Bạn có thể tải file CSV mẫu "
            "để xem tên các cột cần có."
        )

    else:

        try:
            df_upload = read_uploaded_csv(uploaded_file)

            st.subheader("Dữ liệu đã tải lên")

            m1, m2 = st.columns(2)

            m1.metric(
                "Số dòng",
                len(df_upload)
            )

            m2.metric(
                "Số cột",
                len(df_upload.columns)
            )

            st.dataframe(
                df_upload.head(10),
                width="stretch"
            )

            # -------------------------------------------------
            # DỰ ĐOÁN
            # -------------------------------------------------

            if st.button(
                "⚙ Dự đoán tất cả",
                type="primary",
                width="stretch"
            ):

                X_input = prepare_input_data(
                    df_upload,
                    feature_names
                )

                predictions = selected_model.predict(
                    X_input
                )

                result_df = df_upload.copy()

                result_df["Ma_loi_du_doan"] = predictions

                result_df["Ten_loi_du_doan"] = [
                    translate_label(x)
                    for x in predictions
                ]

                # Tính độ tin cậy nếu mô hình hỗ trợ
                if hasattr(selected_model, "predict_proba"):

                    try:
                        probs = selected_model.predict_proba(
                            X_input
                        )

                        result_df["Do_tin_cay_du_doan"] = (
                            probs.max(axis=1) * 100
                        ).round(2)

                    except Exception:
                        pass

                st.success(
                    f"Đã dự đoán {len(result_df)} mẫu "
                    f"bằng {selected_model_name}."
                )

                # -------------------------------------------------
                # THỐNG KÊ KẾT QUẢ
                # -------------------------------------------------

                st.subheader("Thống kê kết quả")

                count_df = (
                    result_df["Ten_loi_du_doan"]
                    .value_counts()
                    .rename_axis("Loại lỗi")
                    .reset_index(name="Số lượng")
                )

                a, b, c = st.columns(3)

                a.metric(
                    "Tổng số mẫu",
                    len(result_df)
                )

                b.metric(
                    "Số loại lỗi",
                    result_df["Ma_loi_du_doan"].nunique()
                )

                if "Do_tin_cay_du_doan" in result_df.columns:

                    c.metric(
                        "Độ tin cậy trung bình",
                        f"{result_df['Do_tin_cay_du_doan'].mean():.1f}%"
                    )

                else:

                    c.metric(
                        "Mô hình",
                        selected_model_name
                    )

                # -------------------------------------------------
                # BIỂU ĐỒ
                # -------------------------------------------------

                st.subheader("Biểu đồ phân bố loại lỗi")

                chart_df = count_df.sort_values(
                    "Số lượng",
                    ascending=True
                )

                fig = px.bar(
                    chart_df,
                    x="Số lượng",
                    y="Loại lỗi",
                    orientation="h",
                    color="Loại lỗi",
                    color_discrete_sequence=PALETTE,
                    text="Số lượng"
                )

                fig.update_traces(
                    textposition="outside",
                    cliponaxis=False
                )

                fig.update_layout(
                    xaxis_title="Số lượng",
                    yaxis_title="",
                    showlegend=False
                )

                st.plotly_chart(
                    style_fig(
                        fig,
                        max(280, 60 * len(chart_df) + 80)
                    ),
                    width="stretch"
                )

                # -------------------------------------------------
                # BẢNG KẾT QUẢ
                # -------------------------------------------------

                st.subheader("Bảng kết quả dự đoán")

                st.dataframe(
                    result_df,
                    width="stretch",
                    hide_index=True
                )

                # -------------------------------------------------
                # TẢI KẾT QUẢ
                # -------------------------------------------------

                st.download_button(
                    "⬇ Tải kết quả dự đoán CSV",
                    data=result_df.to_csv(
                        index=False,
                        encoding="utf-8-sig"
                    ),
                    file_name="ket_qua_du_doan_loi_thep.csv",
                    mime="text/csv",
                    width="stretch"
                )

        except Exception as exc:

            st.error(
                f"Không thể xử lý file CSV: {exc}"
            )

            st.info(
                "Kiểm tra file có đủ 27 cột đặc trưng, "
                "dữ liệu dạng số và không có ô trống. "
                "Hãy đối chiếu với file CSV mẫu."
            )


# =========================================================
# 11. TRANG SO SÁNH MÔ HÌNH
# SO SÁNH 5 MÔ HÌNH TRÊN FILE CSV DO NGƯỜI DÙNG TẢI LÊN
# =========================================================

elif page == "So sánh mô hình":

    st.subheader("📈 So sánh hiệu quả các mô hình")

    st.info(
        "Tải lên file CSV mới để hệ thống sử dụng cùng một "
        "bộ dữ liệu đầu vào và so sánh kết quả của 5 mô hình."
    )

    # -----------------------------------------------------
    # 1. TẢI FILE CSV
    # -----------------------------------------------------

    st.markdown("### 📂 Tải dữ liệu CSV mới")

    uploaded_file = st.file_uploader(
        "Chọn file CSV từ máy tính",
        type=["csv"],
        key="comparison_csv"
    )

    if uploaded_file is None:

        st.warning(
            "Vui lòng tải lên một file CSV để bắt đầu so sánh."
        )

        st.stop()

    # -----------------------------------------------------
    # 2. ĐỌC FILE CSV
    # -----------------------------------------------------

    try:

        try:
            uploaded_df = pd.read_csv(
                uploaded_file,
                encoding="utf-8-sig"
            )

        except UnicodeDecodeError:
            uploaded_file.seek(0)

            uploaded_df = pd.read_csv(
                uploaded_file,
                encoding="latin1"
            )

        uploaded_df.columns = (
            uploaded_df.columns.astype(str).str.strip()
        )

    except Exception as e:

        st.error(f"Không thể đọc file CSV: {e}")
        st.stop()

    if uploaded_df.empty:

        st.error("File CSV không có dữ liệu.")
        st.stop()

    st.success(
        f"Đã tải thành công {len(uploaded_df)} dòng dữ liệu."
    )

    # -----------------------------------------------------
    # 3. KIỂM TRA CẤU TRÚC DỮ LIỆU
    # -----------------------------------------------------

    missing_features = [
        col for col in feature_names
        if col not in uploaded_df.columns
    ]

    if missing_features:

        st.error(
            "File CSV thiếu các đặc trưng cần thiết:"
        )

        st.write(missing_features)

        st.info(
            "File cần có đầy đủ 27 đặc trưng giống dữ liệu "
            "Steel Plates Faults đã sử dụng để huấn luyện."
        )

        st.stop()

    has_actual_label = "Loai_loi" in uploaded_df.columns

    # Chỉ lấy 27 đặc trưng đầu vào
    X_new = uploaded_df[feature_names].copy()

    # Chuyển dữ liệu về dạng số
    X_new = X_new.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X_new = X_new.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Loại bỏ các dòng có đặc trưng không hợp lệ
    valid_mask = X_new.notna().all(axis=1)

    if has_actual_label:

        valid_mask = (
            valid_mask
            & uploaded_df["Loai_loi"].notna()
        )

    X_new = X_new.loc[valid_mask].reset_index(drop=True)

    if has_actual_label:

        y_new = (
            uploaded_df.loc[valid_mask, "Loai_loi"]
            .astype(str)
            .str.strip()
            .reset_index(drop=True)
        )

    if len(X_new) == 0:

        st.error(
            "Không có dòng dữ liệu hợp lệ để so sánh."
        )

        st.stop()

    st.markdown("### 📋 Thông tin dữ liệu")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Số dòng ban đầu",
            len(uploaded_df)
        )

    with col2:
        st.metric(
            "Số dòng hợp lệ",
            len(X_new)
        )

    with col3:
        st.metric(
            "Số đặc trưng",
            len(feature_names)
        )

    if len(X_new) < len(uploaded_df):

        st.warning(
            f"Đã bỏ qua {len(uploaded_df) - len(X_new)} "
            "dòng có dữ liệu không hợp lệ hoặc thiếu nhãn."
        )

    with st.expander("Xem trước dữ liệu CSV"):

        preview_df = X_new.copy()

        if has_actual_label:
            preview_df["Loai_loi"] = y_new

        st.dataframe(
            preview_df.head(10),
            width="stretch"
        )

    # -----------------------------------------------------
    # 4. DỰ ĐOÁN BẰNG CẢ 5 MÔ HÌNH
    # -----------------------------------------------------

    st.markdown("### 🤖 Kết quả dự đoán của 5 mô hình")

    prediction_results = {}
    prediction_errors = []

    progress = st.progress(0)

    model_items = list(models.items())

    for index, (model_name, model) in enumerate(model_items):

        try:

            predictions = model.predict(X_new)

            prediction_results[model_name] = np.asarray(
                predictions
            ).astype(str)

        except Exception as e:

            prediction_errors.append(
                f"{model_name}: {str(e)}"
            )

        progress.progress(
            (index + 1) / len(model_items)
        )

    progress.empty()

    if prediction_errors:

        st.warning(
            "Một số mô hình không thể dự đoán:"
        )

        for error in prediction_errors:
            st.write(error)

    if len(prediction_results) == 0:

        st.error(
            "Không có mô hình nào dự đoán thành công."
        )

        st.stop()

    st.success(
        f"Có {len(prediction_results)}/5 mô hình "
        "dự đoán thành công trên file CSV."
    )

    # -----------------------------------------------------
    # 5. TRƯỜNG HỢP CSV CÓ NHÃN THỰC TẾ
    # -----------------------------------------------------

    if has_actual_label:

        st.markdown("### 🏆 So sánh các chỉ số đánh giá")

        rows = []

        for model_name, predictions in prediction_results.items():

            rows.append({

                "Mô hình": model_name,

                "Accuracy": accuracy_score(
                    y_new,
                    predictions
                ),

                "Precision Macro": precision_score(
                    y_new,
                    predictions,
                    average="macro",
                    zero_division=0
                ),

                "Recall Macro": recall_score(
                    y_new,
                    predictions,
                    average="macro",
                    zero_division=0
                ),

                "F1 Macro": f1_score(
                    y_new,
                    predictions,
                    average="macro",
                    zero_division=0
                )

            })

        comparison_df = pd.DataFrame(rows)

        comparison_df = comparison_df.sort_values(
            "F1 Macro",
            ascending=False
        ).reset_index(drop=True)

        best = comparison_df.iloc[0]

        # Mô hình tốt nhất

        st.markdown(
            "### 🥇 Mô hình có F1 Macro cao nhất"
        )

        st.markdown(
            f"""
            <div class="success-card">
                <h2>{html.escape(best["Mô hình"])}</h2>
                <h3>F1 Macro: {best["F1 Macro"]:.4f}</h3>
                <p>
                    Đây là mô hình có F1 Macro cao nhất
                    trên file CSV vừa tải lên.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Bảng so sánh

        st.markdown("### Bảng so sánh các chỉ số")

        display_df = comparison_df.copy()

        for col in [
            "Accuracy",
            "Precision Macro",
            "Recall Macro",
            "F1 Macro"
        ]:

            display_df[col] = display_df[col].round(4)

        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True
        )

        # Biểu đồ

        st.markdown("### Biểu đồ so sánh")

        metric = st.selectbox(
            "Chọn chỉ số đánh giá",
            [
                "Accuracy",
                "Precision Macro",
                "Recall Macro",
                "F1 Macro"
            ],
            index=3
        )

        fig = px.bar(
            comparison_df,
            x="Mô hình",
            y=metric,
            color="Mô hình",
            text=metric,
            title=f"So sánh các mô hình theo {metric}"
        )

        fig.update_layout(
            yaxis_range=[0, 1],
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        # =========================================================
        # GIẢI THÍCH MÔ HÌNH TỐT NHẤT THEO TỪNG LOẠI LỖI
        # =========================================================

        st.markdown("---")
        st.markdown("### 🧠 Giải thích mô hình dự đoán tốt nhất theo từng loại lỗi")

        # Lấy nhãn thực tế từ file CSV đã tải lên
        # uploaded_df là DataFrame chứa dữ liệu CSV
        actual_labels = (
            uploaded_df["Loai_loi"]
            .astype(str)
            .str.strip()
            .reset_index(drop=True)
        )

        # Tạo bảng chứa dự đoán của tất cả mô hình
        prediction_matrix = pd.DataFrame(
            prediction_results
        ).reset_index(drop=True)

        # Kiểm tra số lượng mẫu
        if len(actual_labels) != len(prediction_matrix):

            st.error(
                "Số lượng nhãn thực tế không khớp với số lượng "
                "kết quả dự đoán. Vui lòng kiểm tra lại dữ liệu."
            )

        else:

            class_results = []

            # Duyệt qua từng loại lỗi thực tế
            for class_name in sorted(actual_labels.unique()):

                # Chọn các mẫu thực tế thuộc loại lỗi hiện tại
                class_mask = actual_labels == class_name

                true_class = actual_labels[class_mask].to_numpy()

                model_scores = {}

                # Tính tỷ lệ dự đoán đúng của từng mô hình
                for model_name in prediction_matrix.columns:

                    predicted_class = (
                        prediction_matrix.loc[class_mask, model_name]
                        .astype(str)
                        .str.strip()
                        .to_numpy()
                    )

                    accuracy = (
                        (predicted_class == true_class).mean() * 100
                    )

                    model_scores[model_name] = accuracy

                # Tìm mô hình có tỷ lệ dự đoán đúng cao nhất
                best_score = max(model_scores.values())

                best_models = [
                    name
                    for name, score in model_scores.items()
                    if abs(score - best_score) < 1e-9
                ]

                class_results.append({

                    "Loại lỗi": class_name,

                    "Số mẫu thực tế": int(class_mask.sum()),

                    "Mô hình tốt nhất": ", ".join(best_models),

                    "Tỷ lệ dự đoán đúng (%)": round(best_score, 2)

                })

            # Tạo bảng kết quả
            class_results_df = pd.DataFrame(class_results)

            st.dataframe(
                class_results_df,
                width="stretch",
                hide_index=True
            )

            st.markdown("#### 🔎 Phân tích kết quả từng loại lỗi")

            # Giải thích theo đặc điểm của thuật toán
            def explain_best_model(model_name):

                name = model_name.lower()

                if "knn" in name or "k-nearest" in name:

                    return (
                        "KNN dự đoán dựa trên các mẫu dữ liệu gần nhất. "
                        "Kết quả cao có thể cho thấy các mẫu thuộc loại lỗi "
                        "này có đặc trưng gần với những mẫu cùng lớp "
                        "trong không gian đặc trưng."
                    )

                elif "random forest" in name or "random_forest" in name:

                    return (
                        "Random Forest kết hợp kết quả từ nhiều cây quyết định. "
                        "Việc tổng hợp nhiều cây có thể giúp mô hình nhận diện "
                        "các mối quan hệ giữa những đặc trưng và giảm sự "
                        "phụ thuộc vào một cây quyết định riêng lẻ."
                    )

                elif "extra trees" in name or "extra_trees" in name:

                    return (
                        "Extra Trees sử dụng nhiều cây quyết định với "
                        "các ngưỡng phân chia ngẫu nhiên. Sự kết hợp này "
                        "có thể giúp mô hình nhận diện các kiểu đặc trưng "
                        "khác nhau của loại lỗi."
                    )

                elif "gradient" in name or "boosting" in name:

                    return (
                        "Gradient Boosting xây dựng các cây quyết định "
                        "theo tuần tự, trong đó mô hình sau tập trung cải thiện "
                        "những sai sót của mô hình trước. Điều này có thể "
                        "giúp tăng khả năng phân loại loại lỗi đang xét."
                    )

                elif "svm" in name:

                    return (
                        "SVM tìm ranh giới phân chia giữa các lớp dữ liệu. "
                        "Khả năng tạo ranh giới phân loại phù hợp, đặc biệt "
                        "với kernel RBF, có thể giúp nhận diện loại lỗi này."
                    )

                return (
                    "Mô hình có thể học được những đặc trưng phù hợp "
                    "để phân biệt loại lỗi đang xét."
                )

            # Hiển thị giải thích cho từng loại lỗi
            for result in class_results:

                class_name = result["Loại lỗi"]
                best_model = result["Mô hình tốt nhất"]
                accuracy = result["Tỷ lệ dự đoán đúng (%)"]
                sample_count = result["Số mẫu thực tế"]

                st.markdown(
                    f"**Đối với lỗi `{class_name}`, mô hình "
                    f"`{best_model}` có tỷ lệ dự đoán đúng cao nhất: "
                    f"{accuracy:.2f}%.**"
                )

                st.write(
                    f"- Số mẫu thuộc loại lỗi này: {sample_count} mẫu."
                )

                st.write(
                    f"- Giải thích: {explain_best_model(best_model)}"
                )

                st.divider()

            st.caption(
                "Lưu ý: Tỷ lệ dự đoán đúng theo từng loại lỗi được tính "
                "bằng số mẫu của lớp được dự đoán chính xác chia cho "
                "tổng số mẫu thực tế thuộc lớp đó. Nguyên nhân giải thích "
                "dựa trên cơ chế thuật toán và mang tính tham khảo; "
                "cần phân tích thêm dữ liệu để xác nhận."
            )

        # -------------------------------------------------
        # 6. SO SÁNH THEO TỪNG LOẠI LỖI
        # -------------------------------------------------

        st.markdown("### 📊 Kết quả theo từng loại lỗi")

        class_rows = []

        for model_name, predictions in prediction_results.items():

            for label in sorted(y_new.unique()):

                mask = (
                    y_new.to_numpy() == label
                )

                total = int(mask.sum())

                correct = int(
                    (
                        predictions[mask]
                        == y_new.to_numpy()[mask]
                    ).sum()
                )

                class_rows.append({

                    "Mô hình": model_name,

                    "Loại lỗi": label,

                    "Tên lỗi": translate_label(label),

                    "Tổng số mẫu": total,

                    "Dự đoán đúng": correct,

                    "Tỷ lệ đúng (%)": round(
                        correct / total * 100,
                        2
                    ) if total else 0

                })

        class_results = pd.DataFrame(class_rows)

        st.dataframe(
            class_results,
            width="stretch",
            hide_index=True
        )

        pivot = class_results.pivot(
            index="Mô hình",
            columns="Tên lỗi",
            values="Tỷ lệ đúng (%)"
        )

        st.markdown("### Bảng tỷ lệ dự đoán đúng theo loại lỗi")

        st.dataframe(
            pivot,
            width="stretch"
        )

    
        # =========================================================
        # 7. SO SÁNH MỨC ĐỘ THỐNG NHẤT DỰ ĐOÁN
        # =========================================================

        st.markdown("### 🔍 So sánh mức độ thống nhất dự đoán")

        if len(prediction_results) >= 2:

            # Tạo DataFrame chứa kết quả dự đoán của các mô hình
            prediction_matrix = pd.DataFrame(
                prediction_results
            )

            agreement_rows = []

            # Duyệt qua từng mô hình
            for model_name in prediction_matrix.columns:

                # Lấy danh sách các mô hình còn lại
                other_models = [
                    col for col in prediction_matrix.columns
                    if col != model_name
                ]

                # So sánh dự đoán của mô hình hiện tại
                # với tất cả các mô hình còn lại
                current_predictions = (
                    prediction_matrix[model_name].to_numpy()[:, None]
                )

                other_predictions = (
                    prediction_matrix[other_models].to_numpy()
                )

                # Đếm số mô hình có dự đoán giống mô hình hiện tại
                same_count = (
                    other_predictions == current_predictions
                ).sum(axis=1)

                # Tính tỷ lệ thống nhất trung bình
                agreement_rate = (
                    same_count / len(other_models)
                ).mean() * 100

                agreement_rows.append({

                    "Mô hình": model_name,

                    "Tỷ lệ thống nhất trung bình (%)": round(
                        agreement_rate, 2
                    )

                })

            # Tạo bảng kết quả
            agreement_df = pd.DataFrame(
                agreement_rows
            ).sort_values(
                "Tỷ lệ thống nhất trung bình (%)",
                ascending=False
            ).reset_index(drop=True)

            st.dataframe(
                agreement_df,
                width="stretch",
                hide_index=True
            )

            # Biểu đồ tỷ lệ thống nhất
            fig_agreement = px.bar(
                agreement_df,
                x="Mô hình",
                y="Tỷ lệ thống nhất trung bình (%)",
                color="Mô hình",
                text="Tỷ lệ thống nhất trung bình (%)",
                title="Mức độ thống nhất dự đoán giữa các mô hình"
            )

            fig_agreement.update_traces(
                texttemplate="%{text:.2f}%",
                textposition="outside"
            )

            fig_agreement.update_layout(
                yaxis_title="Tỷ lệ thống nhất (%)",
                xaxis_title="Mô hình",
                yaxis_range=[0, 110],
                showlegend=False
            )

            st.plotly_chart(
                fig_agreement,
                width="stretch"
            )

            st.caption(
                "Tỷ lệ thống nhất thể hiện mức độ dự đoán trùng khớp "
                "với các mô hình còn lại, không phải tỷ lệ dự đoán đúng."
            )

        else:

            st.warning(
                "Cần ít nhất 2 mô hình để thực hiện so sánh."
            )

    # -----------------------------------------------------
    # 8. TẢI KẾT QUẢ DỰ ĐOÁN
    # -----------------------------------------------------

    st.divider()

    st.markdown("### 📥 Tải kết quả dự đoán")

    output_df = X_new.copy()

    if has_actual_label:
        output_df["Loai_loi_thuc_te"] = y_new

    for model_name, predictions in prediction_results.items():

        output_df[
            f"Du_doan_{model_name.replace(' ', '_')}"
        ] = predictions

    csv_data = output_df.to_csv(
        index=False,
        encoding="utf-8-sig"
    )

    st.download_button(
        label="⬇️ Tải kết quả so sánh CSV",
        data=csv_data,
        file_name="ket_qua_so_sanh_mo_hinh.csv",
        mime="text/csv"
    )

    # -----------------------------------------------------
    # SO SÁNH THEO TỪNG LOẠI LỖI
    # -----------------------------------------------------

    st.markdown(
        "### So sánh mô hình theo từng loại lỗi"
    )

    class_results = get_predictions_by_class()

    if not class_results.empty:

        selected_class = st.selectbox(
            "Chọn loại lỗi cần so sánh",

            sorted(
                class_results["Loại lỗi"].unique()
            ),

            format_func=lambda x: (
                f"{x} - {translate_label(x)}"
            ),

            key="compare_class"
        )

        class_df = class_results[
            class_results["Loại lỗi"] == selected_class
        ].sort_values(
            "Tỷ lệ đúng (%)",
            ascending=False
        )

        st.write(
            f"**Loại lỗi đang xem:** "
            f"{selected_class} - "
            f"{translate_label(selected_class)}"
        )

        st.dataframe(
            class_df[
                [
                    "Mô hình",
                    "Tổng số mẫu",
                    "Dự đoán đúng",
                    "Tỷ lệ đúng (%)"
                ]
            ],
            width="stretch",
            hide_index=True
        )

        best_class_model = class_df.iloc[0]

        st.success(
            f"Đối với lỗi {selected_class}, mô hình "
            f"{best_class_model['Mô hình']} có tỷ lệ "
            f"dự đoán đúng cao nhất: "
            f"{best_class_model['Tỷ lệ đúng (%)']:.2f}%."
        )

        fig_class = px.bar(
            class_df,
            x="Mô hình",
            y="Tỷ lệ đúng (%)",
            color="Mô hình",
            text="Tỷ lệ đúng (%)",
            title=f"Tỷ lệ dự đoán đúng - {selected_class}"
        )

        fig_class.update_layout(
            yaxis_range=[0, 100],
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(
            fig_class,
            width="stretch"
        )

    # -----------------------------------------------------
    # BẢNG TỔNG HỢP CÁC LOẠI LỖI
    # -----------------------------------------------------

    st.markdown(
        "### Tổng hợp khả năng nhận dạng từng loại lỗi"
    )

    st.caption(
        "Bảng thể hiện tỷ lệ dự đoán đúng của từng mô hình "
        "trên từng loại lỗi trong file dữ liệu."
    )

    if not class_results.empty:

        pivot = class_results.pivot(
            index="Mô hình",
            columns="Loại lỗi",
            values="Tỷ lệ đúng (%)"
        )

        st.dataframe(
            pivot.round(2),
            width="stretch"
        )


# =========================================================
# 12. TRANG THÔNG TIN HỆ THỐNG
# =========================================================

elif page == "Thông tin hệ thống":

    st.subheader(
        "⚙️ Thông tin hệ thống"
    )

    st.markdown(
        "### Dữ liệu sử dụng"
    )

    st.write(
        f"**Số mẫu hợp lệ:** {len(X_all)}"
    )

    st.write(
        f"**Số đặc trưng:** {len(feature_names)}"
    )

    st.write(
        f"**Số loại lỗi:** {y_all.nunique()}"
    )

    st.markdown(
        "### Các mô hình học máy"
    )

    for name, filename in MODEL_FILES.items():

        if name in models:

            st.success(
                f"{name}: Đã tải thành công"
            )

        else:

            st.error(
                f"{name}: Chưa tải được ({filename})"
            )

    st.markdown(
        "### Danh sách 27 đặc trưng"
    )

    for group_name, columns in FEATURE_GROUPS.items():

        st.write(
            f"**{group_name}:** {', '.join(columns)}"
        )


    if model_errors:

        with st.expander(
            "Chi tiết các tệp mô hình chưa tải được"
        ):

            for error in model_errors:
                st.write(error)


