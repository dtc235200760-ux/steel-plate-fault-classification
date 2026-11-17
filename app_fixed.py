import os
import html
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ============================================================
# 1. CẤU HÌNH TRANG VÀ GIAO DIỆN
# ============================================================

st.set_page_config(
    page_title="Phân loại lỗi bề mặt tấm thép",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

INK = "#14202b"
MUTED = "#64748b"
LINE = "#d9e0e6"
BLUE = "#2563eb"
ACCENT = "#ea580c"
PALETTE = ["#2563eb", "#ea580c", "#16a34a", "#8b5cf6", "#0891b2", "#db2777", "#ca8a04"]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&display=swap');

:root {
    --ink:#14202b; --muted:#64748b; --line:#d9e0e6;
    --bg:#f3f6fa; --surface:#ffffff; --blue:#2563eb;
    --accent:#ea580c; --graphite:#14202b;
}
html, body, [class*="css"], .stApp {
    font-family:'Be Vietnam Pro',sans-serif;
}
[data-testid="stAppViewContainer"] {background:var(--bg); color:var(--ink);}
[data-testid="stHeader"] {background:transparent;}
.block-container {padding-top:1.7rem; padding-bottom:3rem; max-width:1500px;}
h1,h2,h3,h4 {color:var(--ink); letter-spacing:-.25px;}
h3 {font-weight:700 !important;}
p,label {color:#334155;}
section[data-testid="stSidebar"] {background:#14202b;}
section[data-testid="stSidebar"] * {color:#e6edf5;}
section[data-testid="stSidebar"] > div {padding-top:1.2rem;}
.brand {
    padding:8px 5px 18px; margin-bottom:14px;
    border-bottom:1px solid rgba(255,255,255,.15);
}
.brand-name {font-size:27px;font-weight:800;letter-spacing:-.5px;}
.brand-sub {font-size:12px;color:#a9bacb !important;line-height:1.5;}
.side-foot {font-size:12px;color:#9fb0c0 !important;line-height:1.7;padding:12px 5px;}
section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding:10px 12px;border-radius:8px;border-left:3px solid transparent;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background:rgba(255,255,255,.08);
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background:rgba(255,255,255,.12);border-left-color:#fb923c;font-weight:700;
}
.page-head {
    margin:0 0 1.25rem;padding:0 0 1rem;
    border-bottom:1px solid var(--line);
}
.page-head h1 {font-size:clamp(25px,2.5vw,35px);font-weight:800;margin:0 0 7px;}
.page-head p {color:var(--muted);font-size:14px;margin:0;line-height:1.65;}
/* Dải trang trí ngang nhiều màu, không sử dụng ảnh */
.color-strip {
    display:flex;gap:5px;height:9px;width:100%;
    margin:8px 0 24px;border-radius:20px;overflow:hidden;
}
.color-strip span {display:block;flex:1;border-radius:20px;}
.hero-band {
    position:relative;overflow:hidden;border-radius:14px;
    padding:25px 28px;margin:8px 0 22px;color:white;
    background:linear-gradient(105deg,#14202b 0%,#1e4975 42%,#2563eb 72%,#0f9f91 100%);
}
.hero-band:after {
    content:"";position:absolute;width:270px;height:270px;border-radius:50%;
    right:5%;top:-170px;border:35px solid rgba(255,255,255,.10);
}
.hero-band h2 {color:white;font-size:25px;margin:0 0 7px;font-weight:800;}
.hero-band p {color:#e4eefb;margin:0;max-width:850px;font-size:14px;line-height:1.7;}
.hero-label {
    display:inline-block;padding:5px 10px;border-radius:20px;
    background:rgba(255,255,255,.15);color:white;font-size:11px;
    font-weight:700;letter-spacing:.7px;margin-bottom:10px;
}
div[data-testid="stMetric"] {
    background:white;border:1px solid var(--line);border-radius:10px;padding:15px 17px;
}
div[data-testid="stMetric"] label p {color:var(--muted) !important;font-weight:600;}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {color:var(--ink);font-weight:800;}
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color:var(--line) !important;border-radius:10px;background:white;
}
[data-testid="stPlotlyChart"],[data-testid="stDataFrame"],[data-testid="stTable"] {
    background:white;border:1px solid var(--line);border-radius:10px;overflow:hidden;
}
[data-testid="stPlotlyChart"] {padding:5px;}
[data-testid="stAlert"] {border-radius:9px;}
div[data-baseweb="input"] > div,div[data-baseweb="select"] > div {
    border-radius:7px !important;background:white !important;
}
button[data-baseweb="tab"] {font-weight:600;padding:10px 16px;}
button[data-baseweb="tab"][aria-selected="true"] {color:var(--accent) !important;}
[data-baseweb="tab-highlight"] {background:var(--accent) !important;}
.stButton > button,.stDownloadButton > button,[data-testid="stFormSubmitButton"] > button {
    border-radius:8px;min-height:43px;font-weight:700;
}
.stButton > button[kind="primary"],[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-primaryFormSubmit"] {
    background:var(--accent);border-color:var(--accent);color:white;
}
.stButton > button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover,
[data-testid="stBaseButton-primaryFormSubmit"]:hover {
    background:#c2410c;border-color:#c2410c;color:white;
}
.result-tag {
    background:white;border:1px solid var(--line);border-left:6px solid var(--accent);
    border-radius:10px;padding:20px 24px;margin:8px 0 18px;
}
.result-tag .tag-label {color:var(--muted);font-size:13px;margin:0 0 5px;}
.result-tag .tag-name {color:var(--ink);font-size:clamp(24px,3vw,34px);font-weight:800;margin:0;}
.result-tag .tag-meta {color:var(--muted);font-size:13px;margin:8px 0 0;}
.meter-row {display:flex;justify-content:space-between;margin:17px 0 6px;color:var(--muted);font-size:13px;}
.meter-row strong {font-size:17px;color:var(--ink);}
.meter {height:8px;background:#e7edf3;border-radius:10px;overflow:hidden;}
.meter-fill {height:100%;background:linear-gradient(90deg,#2563eb,#0d9488,#f97316);border-radius:10px;}
.chips {display:flex;flex-wrap:wrap;gap:8px;margin:8px 0 15px;}
.chip {background:white;border:1px solid var(--line);border-radius:30px;padding:7px 13px;font-size:13px;color:var(--ink);}
.chip:before {content:"";display:inline-block;width:8px;height:8px;border-radius:50%;background:#16a34a;margin-right:8px;}
.chip.miss:before {background:#d97706;}
@media(max-width:768px) {
 .block-container {padding:1rem .8rem 2rem;}
 .hero-band {padding:20px;}
}
</style>
""", unsafe_allow_html=True)


def color_strip():
    colors = ["#2563eb", "#0d9488", "#16a34a", "#eab308", "#f97316", "#db2777", "#8b5cf6"]
    spans = "".join(f'<span style="background:{c}"></span>' for c in colors)
    st.markdown(f'<div class="color-strip">{spans}</div>', unsafe_allow_html=True)


def page_head(title, description):
    st.markdown(
        f'<div class="page-head"><h1>{html.escape(title)}</h1>'
        f'<p>{html.escape(description)}</p></div>',
        unsafe_allow_html=True
    )
    color_strip()


def hero(title, description, label="STEEL AI • QUALITY INSPECTION"):
    st.markdown(
        f'<div class="hero-band"><div class="hero-label">{label}</div>'
        f'<h2>{html.escape(title)}</h2><p>{html.escape(description)}</p></div>',
        unsafe_allow_html=True
    )


# ============================================================
# 2. ĐƯỜNG DẪN VÀ CẤU HÌNH MÔ HÌNH
# ============================================================

MODEL_DIR = "models"
RESULT_DIR = "results"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

MODEL_FILES = {
    "SVM": "svm.joblib",
    "KNN": "knn.joblib",
    "Random Forest": "random_forest.joblib",
    "Extra Trees": "extra_trees.joblib",
    "Gradient Boosting": "gradient_boosting.joblib"
}

CLASS_TRANSLATIONS = {
    "Bumps": "Lỗi dạng lồi, gồ ghề",
    "Dirtiness": "Lỗi bề mặt bẩn",
    "K_Scratch": "Lỗi vết xước dạng K",
    "Other_Faults": "Các lỗi khác",
    "Pastry": "Lỗi dạng vết khuyết không đều",
    "Stains": "Lỗi vết ố, vết bẩn",
    "Z_Scratch": "Lỗi vết xước dạng Z"
}

FEATURE_GROUPS = {
    "Vị trí": ["X_Minimum", "X_Maximum", "Y_Minimum", "Y_Maximum"],
    "Diện tích và chu vi": ["Pixels_Areas", "X_Perimeter", "Y_Perimeter"],
    "Độ sáng": ["Sum_of_Luminosity", "Minimum_of_Luminosity",
                "Maximum_of_Luminosity", "Luminosity_Index"],
    "Hình dạng và đường biên": [
        "Edges_Index", "Empty_Index", "Square_Index", "Outside_X_Index",
        "Edges_X_Index", "Edges_Y_Index", "Outside_Global_Index", "Orientation_Index"
    ],
    "Biến đổi": ["LogOfAreas", "Log_X_Index", "Log_Y_Index", "SigmoidOfAreas"],
    "Vật liệu và sản xuất": [
        "Length_of_Conveyer", "TypeOfSteel_A300",
        "TypeOfSteel_A400", "Steel_Plate_Thickness"
    ]
}


# ============================================================
# 3. TẢI MÔ HÌNH VÀ ĐẶC TRƯNG
# ============================================================

@st.cache_resource
def load_feature_names():
    path = os.path.join(MODEL_DIR, "feature_names.joblib")
    return joblib.load(path) if os.path.exists(path) else None


@st.cache_resource
def load_models():
    loaded, errors = {}, {}
    for name, filename in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            try:
                loaded[name] = joblib.load(path)
            except Exception as exc:
                errors[name] = str(exc)
    return loaded, errors


def load_best_model_name():
    path = os.path.join(MODEL_DIR, "best_model_name.joblib")
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None


feature_names = load_feature_names()
models, model_errors = load_models()
best_model_name = load_best_model_name()


def translate_class(label):
    return CLASS_TRANSLATIONS.get(str(label), str(label))


def predict_one(model, input_data):
    prediction = model.predict(input_data)[0]
    probabilities = None
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(input_data)[0]
        except Exception:
            pass
    return prediction, probabilities


def default_model_index():
    names = list(models.keys())
    return names.index(best_model_name) if best_model_name in names else 0


# ============================================================
# 4. XỬ LÝ FILE CSV VÀ DỮ LIỆU
# ============================================================

def read_uploaded_csv(uploaded_file):
    uploaded_file.seek(0)
    try:
        df = pd.read_csv(uploaded_file, sep=None, engine="python", encoding="utf-8-sig")
        if df.shape[1] > 1:
            return df
    except Exception:
        pass

    uploaded_file.seek(0)
    try:
        df = pd.read_csv(uploaded_file, sep=",", encoding="utf-8-sig")
        if df.shape[1] > 1:
            return df
    except Exception:
        pass

    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file, sep=";", encoding="utf-8-sig")


def prepare_input_data(df, names):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    missing = [col for col in names if col not in df.columns]
    if missing:
        raise ValueError("File thiếu các đặc trưng: " + ", ".join(missing))

    X = df[names].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    if X.isnull().any().any():
        raise ValueError("Dữ liệu có ô trống hoặc giá trị không phải số.")
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Dữ liệu chứa giá trị vô cực hoặc không hợp lệ.")
    return X


def group_features(names):
    groups, assigned = {}, set()
    for group_name, group_cols in FEATURE_GROUPS.items():
        present = [f for f in names if f in group_cols]
        if present:
            groups[group_name] = present
            assigned.update(present)
    others = [f for f in names if f not in assigned]
    if others:
        groups["Khác"] = others
    return groups


def render_feature_inputs(names, key_prefix):
    groups = group_features(names)
    tabs = st.tabs([f"{name} ({len(cols)})" for name, cols in groups.items()])
    values = {}
    for tab, (group_name, cols) in zip(tabs, groups.items()):
        with tab:
            columns = st.columns(3)
            for i, feature in enumerate(cols):
                with columns[i % 3]:
                    values[feature] = st.number_input(
                        feature, value=0.0, format="%.6f",
                        key=f"{key_prefix}_{feature}"
                    )
    return values


# ============================================================
# 5. BIỂU ĐỒ VÀ THÀNH PHẦN HIỂN THỊ
# ============================================================

def style_fig(fig, height=400):
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Be Vietnam Pro, sans-serif", size=13, color=INK),
        margin=dict(l=12, r=18, t=18, b=12),
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        colorway=PALETTE,
        legend=dict(orientation="h", y=-0.22, x=0, title_text="")
    )
    fig.update_xaxes(gridcolor="#e6ebf0", zeroline=False)
    fig.update_yaxes(gridcolor="#e6ebf0", zeroline=False)
    return fig


def result_tag(prediction, model_name, confidence=None):
    meter = ""
    if confidence is not None:
        safe_confidence = min(max(float(confidence), 0), 100)
        meter = (
            '<div class="meter-row"><span>Độ tin cậy dự đoán</span>'
            f'<strong>{safe_confidence:.1f}%</strong></div>'
            '<div class="meter"><div class="meter-fill" '
            f'style="width:{safe_confidence:.1f}%"></div></div>'
        )
    st.markdown(
        '<div class="result-tag"><p class="tag-label">Loại lỗi dự đoán</p>'
        f'<p class="tag-name">{html.escape(translate_class(prediction))}</p>'
        f'<p class="tag-meta">Mã lỗi <b>{html.escape(str(prediction))}</b> · '
        f'Mô hình <b>{html.escape(model_name)}</b></p>{meter}</div>',
        unsafe_allow_html=True
    )


def probability_chart(classes, probabilities):
    df = pd.DataFrame({
        "Loại lỗi": [translate_class(c) for c in classes],
        "Xác suất (%)": np.asarray(probabilities) * 100
    }).sort_values("Xác suất (%)", ascending=True)

    colors = ["#94a3b8"] * len(df)
    if colors:
        colors[-1] = ACCENT

    fig = px.bar(df, x="Xác suất (%)", y="Loại lỗi", orientation="h")
    fig.update_traces(
        marker_color=colors, texttemplate="%{x:.1f}%",
        textposition="outside", cliponaxis=False
    )
    fig.update_layout(
        xaxis_title="Xác suất (%)", yaxis_title="",
        xaxis_range=[0, 112], showlegend=False
    )
    return style_fig(fig, 360), df.sort_values("Xác suất (%)", ascending=False)


# ============================================================
# 6. THANH ĐIỀU HƯỚNG
# ============================================================

st.sidebar.markdown(
    '<div class="brand"><div class="brand-name">🏭 Steel AI</div>'
    '<div class="brand-sub">PHÂN LOẠI LỖI BỀ MẶT TẤM THÉP<br>'
    'Hệ thống hỗ trợ kiểm tra chất lượng</div></div>',
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Điều hướng",
    ["Tổng quan", "Dự đoán lỗi", "Dự đoán hàng loạt",
     "So sánh mô hình", "Thông tin hệ thống"],
    label_visibility="collapsed"
)

st.sidebar.markdown(
    '<div class="side-foot">UCI Steel Plates Faults<br>'
    f'{len(MODEL_FILES)} mô hình học máy<br>Machine Learning Classification</div>',
    unsafe_allow_html=True
)

if not models:
    st.error("Không tìm thấy mô hình đã huấn luyện trong thư mục models.")
    st.info("Hãy chạy python main.py trước để tạo các file mô hình.")
    st.stop()

if feature_names is None:
    st.error("Không tìm thấy models/feature_names.joblib. Hãy chạy lại main.py.")
    st.stop()


# ============================================================
# TRANG 1: TỔNG QUAN
# ============================================================

if page == "Tổng quan":
    page_head(
        "Phân loại lỗi bề mặt tấm thép",
        "Ứng dụng học máy phân tích các đặc trưng hình học và độ sáng "
        "để hỗ trợ nhận diện 7 loại lỗi trên bề mặt thép."
    )
    hero(
        "STEEL AI — SMART QUALITY INSPECTION",
        "Phân tích dữ liệu đặc trưng của tấm thép, dự đoán loại lỗi và "
        "hỗ trợ đánh giá kết quả của nhiều mô hình học máy."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Đặc trưng đầu vào", len(feature_names))
    c2.metric("Mô hình đã tải", f"{len(models)}/{len(MODEL_FILES)}")
    c3.metric("Loại lỗi", len(CLASS_TRANSLATIONS))

    if best_model_name in models:
        st.info(f"Mô hình được chọn sau huấn luyện: **{best_model_name}**.")

    st.subheader("Trạng thái các mô hình")
    chips = "".join(
        f'<span class="chip{"" if name in models else " miss"}">{html.escape(name)}</span>'
        for name in MODEL_FILES
    )
    st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)

    if model_errors:
        with st.expander("Mô hình không tải được"):
            for name, message in model_errors.items():
                st.write(f"**{name}:** {message}")

    st.subheader("Quy trình sử dụng")
    guide_cols = st.columns(3)
    guides = [
        ("01 · Dự đoán lỗi", "Nhập thông số của một mẫu thép để nhận kết quả dự đoán."),
        ("02 · Dự đoán hàng loạt", "Tải file CSV có nhiều mẫu và xuất kết quả dự đoán."),
        ("03 · So sánh mô hình", "Đối chiếu các chỉ số và kết quả dự đoán của 5 mô hình.")
    ]
    for col, (title, description) in zip(guide_cols, guides):
        with col:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.write(description)

    st.subheader("Danh sách 7 loại lỗi")
    class_df = pd.DataFrame({
        "Mã lỗi": list(CLASS_TRANSLATIONS.keys()),
        "Tên lỗi": list(CLASS_TRANSLATIONS.values())
    })
    st.dataframe(class_df, width="stretch", hide_index=True)


# ============================================================
# TRANG 2: DỰ ĐOÁN LỖI
# ============================================================

elif page == "Dự đoán lỗi":
    page_head(
        "Dự đoán loại lỗi bề mặt",
        "Chọn mô hình và nhập các thông số đặc trưng của một tấm thép."
    )
    hero(
        "PHÂN TÍCH MỘT MẪU THÉP",
        "Nhập đủ các đặc trưng theo nhóm. Hệ thống sẽ đưa dữ liệu vào mô hình "
        "đã huấn luyện và hiển thị loại lỗi dự đoán."
    )

    selected_model_name = st.selectbox(
        "Mô hình dự đoán", list(models.keys()), index=default_model_index()
    )
    selected_model = models[selected_model_name]
    st.subheader(f"Thông số đặc trưng ({len(feature_names)})")
    input_values = render_feature_inputs(feature_names, "input")

    if st.button("🔎 Dự đoán loại lỗi", type="primary", width="stretch"):
        input_df = pd.DataFrame([input_values], columns=feature_names)
        try:
            prediction, probabilities = predict_one(selected_model, input_df)
            confidence = float(np.max(probabilities) * 100) if probabilities is not None else None
            st.subheader("Kết quả dự đoán")
            result_tag(prediction, selected_model_name, confidence)

            if probabilities is not None:
                fig, probability_df = probability_chart(
                    selected_model.classes_, probabilities
                )
                st.markdown("**Xác suất của từng loại lỗi**")
                st.plotly_chart(fig, width="stretch")
                st.dataframe(probability_df, width="stretch", hide_index=True)
            else:
                st.info("Mô hình này không hỗ trợ tính xác suất dự đoán.")
        except Exception as exc:
            st.error(f"Không thể dự đoán: {exc}")


# ============================================================
# TRANG 3: DỰ ĐOÁN HÀNG LOẠT
# ============================================================

elif page == "Dự đoán hàng loạt":
    page_head(
        "Dự đoán lỗi từ file CSV",
        "Tải lên dữ liệu CSV chứa các đặc trưng của nhiều tấm thép."
    )
    hero(
        "BATCH PREDICTION",
        "Hệ thống kiểm tra cấu trúc dữ liệu, dự đoán từng dòng và tổng hợp "
        "số lượng các loại lỗi được nhận diện."
    )

    left, right = st.columns([2, 1])
    with left:
        selected_model_name = st.selectbox(
            "Mô hình dự đoán", list(models.keys()), index=default_model_index(),
            key="batch_model"
        )
    selected_model = models[selected_model_name]
    with right:
        st.write("")
        st.download_button(
            "⬇ Tải file CSV mẫu",
            data=pd.DataFrame(columns=feature_names).to_csv(
                index=False, encoding="utf-8-sig"
            ),
            file_name="mau_dac_trung_thep.csv",
            mime="text/csv",
            width="stretch"
        )

    uploaded_file = st.file_uploader("Chọn file CSV", type=["csv"])
    if uploaded_file is None:
        st.info("Chưa có file. Bạn có thể tải file CSV mẫu để xem tên các cột cần có.")

    if uploaded_file is not None:
        try:
            df = read_uploaded_csv(uploaded_file)
            st.subheader("Dữ liệu đã tải lên")
            m1, m2 = st.columns(2)
            m1.metric("Số dòng", len(df))
            m2.metric("Số cột", len(df.columns))
            st.dataframe(df.head(10), width="stretch")

            if st.button("⚙ Dự đoán tất cả", type="primary", width="stretch"):
                X_input = prepare_input_data(df, feature_names)
                predictions = selected_model.predict(X_input)
                result_df = df.copy()
                result_df["Ma_loi_du_doan"] = predictions
                result_df["Ten_loi_du_doan"] = [
                    translate_class(x) for x in predictions
                ]

                if hasattr(selected_model, "predict_proba"):
                    try:
                        probs = selected_model.predict_proba(X_input)
                        result_df["Do_tin_cay_du_doan"] = (
                            probs.max(axis=1) * 100
                        ).round(2)
                    except Exception:
                        pass

                st.success(
                    f"Đã dự đoán {len(result_df)} mẫu bằng {selected_model_name}."
                )
                st.subheader("Thống kê kết quả")
                count_df = (
                    result_df["Ten_loi_du_doan"].value_counts()
                    .rename_axis("Loại lỗi").reset_index(name="Số lượng")
                )
                a, b, c = st.columns(3)
                a.metric("Tổng số mẫu", len(result_df))
                b.metric("Số loại lỗi", result_df["Ma_loi_du_doan"].nunique())
                if "Do_tin_cay_du_doan" in result_df:
                    c.metric(
                        "Độ tin cậy trung bình",
                        f"{result_df['Do_tin_cay_du_doan'].mean():.1f}%"
                    )
                else:
                    c.metric("Mô hình", selected_model_name)

                chart_df = count_df.sort_values("Số lượng", ascending=True)
                fig = px.bar(
                    chart_df, x="Số lượng", y="Loại lỗi", orientation="h",
                    color="Loại lỗi", color_discrete_sequence=PALETTE,
                    text="Số lượng"
                )
                fig.update_traces(textposition="outside", cliponaxis=False)
                fig.update_layout(xaxis_title="Số lượng", yaxis_title="", showlegend=False)
                st.plotly_chart(
                    style_fig(fig, max(280, 60 * len(chart_df) + 80)),
                    width="stretch"
                )

                st.subheader("Bảng kết quả")
                st.dataframe(result_df, width="stretch")
                st.download_button(
                    "⬇ Tải kết quả dự đoán CSV",
                    data=result_df.to_csv(index=False, encoding="utf-8-sig"),
                    file_name="ket_qua_du_doan_loi_thep.csv",
                    mime="text/csv",
                    width="stretch"
                )
        except Exception as exc:
            st.error(f"Không thể xử lý file CSV: {exc}")
            st.info(
                "Kiểm tra file có đủ tên cột đặc trưng, dữ liệu dạng số "
                "và không có ô trống. Hãy đối chiếu với file CSV mẫu."
            )


# ============================================================
# TRANG 4: SO SÁNH MÔ HÌNH
# ============================================================

elif page == "So sánh mô hình":
    page_head(
        "So sánh các mô hình học máy",
        "Đánh giá hiệu quả trên tập Test và so sánh dự đoán của các mô hình "
        "trên cùng một mẫu thép."
    )
    hero(
        "MODEL PERFORMANCE COMPARISON",
        "Đối chiếu Accuracy, Precision, Recall, F1-score, ROC-AUC và mức độ "
        "đồng thuận giữa các mô hình."
    )

    tab_metrics, tab_sample = st.tabs([
        "📊 Hiệu quả trên tập Test",
        "🔬 Dự đoán trên cùng một mẫu"
    ])

    with tab_metrics:
        comparison_path = os.path.join(RESULT_DIR, "so_sanh_mo_hinh.csv")
        if os.path.exists(comparison_path):
            try:
                comparison_df = pd.read_csv(comparison_path, encoding="utf-8-sig")
                st.subheader("Bảng chỉ số đánh giá")
                desired = [
                    "Mo_hinh", "Test_Accuracy", "Test_Precision_Macro",
                    "Test_Recall_Macro", "Test_F1_Macro", "Test_AUC_Macro_OVR"
                ]
                available = [col for col in desired if col in comparison_df.columns]
                if not available:
                    st.warning("CSV không có các cột chỉ số Test theo định dạng mong đợi.")
                else:
                    test_df = comparison_df[available].copy()
                    numeric_cols = [c for c in available if c != "Mo_hinh"]
                    for col in numeric_cols:
                        test_df[col] = pd.to_numeric(test_df[col], errors="coerce")
                    test_df = test_df.rename(columns={
                        "Mo_hinh": "Mô hình",
                        "Test_Accuracy": "Accuracy",
                        "Test_Precision_Macro": "Precision Macro",
                        "Test_Recall_Macro": "Recall Macro",
                        "Test_F1_Macro": "F1 Macro",
                        "Test_AUC_Macro_OVR": "ROC-AUC Macro"
                    })
                    shown_numeric = [c for c in test_df.columns if c != "Mô hình"]
                    st.dataframe(
                        test_df.style.format("{:.4f}", subset=shown_numeric)
                        .highlight_max(subset=shown_numeric, color="#ffedcc"),
                        width="stretch", hide_index=True
                    )
                    st.caption("Các ô được tô màu là giá trị cao nhất của từng chỉ số.")

                    st.subheader("Biểu đồ so sánh nhiều chỉ số")
                    metric_map = {
                        "Accuracy": "Test_Accuracy",
                        "Precision Macro": "Test_Precision_Macro",
                        "Recall Macro": "Test_Recall_Macro",
                        "F1 Macro": "Test_F1_Macro",
                        "ROC-AUC Macro": "Test_AUC_Macro_OVR"
                    }
                    metric_options = {
                        label: col for label, col in metric_map.items()
                        if col in comparison_df.columns
                    }
                    selected_metrics = st.multiselect(
                        "Chọn chỉ số muốn hiển thị",
                        list(metric_options.keys()),
                        default=list(metric_options.keys())
                    )
                    if selected_metrics:
                        chart_df = comparison_df[["Mo_hinh"] + [
                            metric_options[m] for m in selected_metrics
                        ]].copy()
                        chart_df = chart_df.rename(columns={
                            metric_options[m]: m for m in selected_metrics
                        })
                        long_df = chart_df.melt(
                            id_vars="Mo_hinh", var_name="Chỉ số", value_name="Giá trị"
                        )
                        long_df["Giá trị"] = pd.to_numeric(
                            long_df["Giá trị"], errors="coerce"
                        )
                        fig = px.bar(
                            long_df, x="Mo_hinh", y="Giá trị", color="Chỉ số",
                            barmode="group", text_auto=".3f",
                            color_discrete_sequence=PALETTE
                        )
                        fig.update_traces(textposition="outside", cliponaxis=False)
                        fig.update_layout(
                            xaxis_title="", yaxis_title="Giá trị",
                            yaxis_range=[0, 1.12]
                        )
                        st.plotly_chart(style_fig(fig, 460), width="stretch")
                    else:
                        st.info("Chọn ít nhất một chỉ số để hiển thị biểu đồ.")

                if "Val_F1_Macro" in comparison_df.columns and "Mo_hinh" in comparison_df.columns:
                    comparison_df["Val_F1_Macro"] = pd.to_numeric(
                        comparison_df["Val_F1_Macro"], errors="coerce"
                    )
                    valid = comparison_df.dropna(subset=["Val_F1_Macro"])
                    if not valid.empty:
                        best_row = valid.loc[valid["Val_F1_Macro"].idxmax()]
                        st.subheader("Mô hình được lựa chọn")
                        x, y = st.columns(2)
                        x.metric("Mô hình tốt nhất", str(best_row["Mo_hinh"]))
                        y.metric("Validation F1-Macro", f"{best_row['Val_F1_Macro']:.4f}")
                        st.caption(
                            "Mô hình được chọn theo F1-Macro trên tập Validation. "
                            "Tập Test được dùng để đánh giá hiệu quả cuối cùng."
                        )
            except Exception as exc:
                st.error(f"Không thể đọc kết quả so sánh: {exc}")
        else:
            st.warning(
                "Chưa có results/so_sanh_mo_hinh.csv. Hãy chạy main.py để tạo file này."
            )

    with tab_sample:
        st.write(
            "Nhập một mẫu thép. Các mô hình đã tải sẽ nhận cùng một bộ đặc trưng "
            "và đưa ra dự đoán để đối chiếu."
        )
        with st.form("compare_prediction_form"):
            compare_values = render_feature_inputs(feature_names, "compare_feature")
            compare_button = st.form_submit_button(
                "So sánh dự đoán của các mô hình",
                type="primary", width="stretch"
            )

        if compare_button:
            input_df = pd.DataFrame([compare_values], columns=feature_names)
            rows = []
            for model_name, model in models.items():
                try:
                    prediction, probabilities = predict_one(model, input_df)
                    confidence = (
                        round(float(np.max(probabilities) * 100), 2)
                        if probabilities is not None else None
                    )
                    rows.append({
                        "Mô hình": model_name,
                        "Mã lỗi dự đoán": str(prediction),
                        "Tên lỗi dự đoán": translate_class(prediction),
                        "Độ tin cậy (%)": confidence,
                        "Trạng thái": "Thành công"
                    })
                except Exception as exc:
                    rows.append({
                        "Mô hình": model_name,
                        "Mã lỗi dự đoán": "-",
                        "Tên lỗi dự đoán": "-",
                        "Độ tin cậy (%)": None,
                        "Trạng thái": f"Lỗi: {exc}"
                    })

            result_df = pd.DataFrame(rows)
            st.subheader("Kết quả dự đoán của từng mô hình")
            st.dataframe(
                result_df, width="stretch", hide_index=True,
                column_config={
                    "Độ tin cậy (%)": st.column_config.ProgressColumn(
                        "Độ tin cậy (%)", format="%.2f%%",
                        min_value=0, max_value=100
                    )
                }
            )

            successful = result_df[result_df["Trạng thái"] == "Thành công"]
            if not successful.empty:
                counts = (
                    successful["Tên lỗi dự đoán"].value_counts()
                    .rename_axis("Loại lỗi").reset_index(name="Số mô hình")
                )
                a, b = st.columns(2)
                a.metric("Mô hình dự đoán thành công", len(successful))
                b.metric("Số loại lỗi được dự đoán", successful["Mã lỗi dự đoán"].nunique())

                most_common = counts.iloc[0]
                if successful["Mã lỗi dự đoán"].nunique() == 1:
                    st.success(f"Tất cả mô hình cùng dự đoán: **{most_common['Loại lỗi']}**.")
                else:
                    st.warning(
                        f"Các mô hình chưa thống nhất. Kết quả được chọn nhiều nhất: "
                        f"**{most_common['Loại lỗi']}** ({int(most_common['Số mô hình'])} mô hình)."
                    )

                fig = px.bar(
                    counts.sort_values("Số mô hình"),
                    x="Số mô hình", y="Loại lỗi", orientation="h",
                    color="Loại lỗi", color_discrete_sequence=PALETTE,
                    text="Số mô hình"
                )
                fig.update_traces(textposition="outside", cliponaxis=False)
                fig.update_layout(xaxis_title="Số mô hình", yaxis_title="", showlegend=False)
                st.plotly_chart(
                    style_fig(fig, max(250, 60 * len(counts) + 80)),
                    width="stretch"
                )
            else:
                st.error("Không có mô hình nào dự đoán thành công.")


# ============================================================
# TRANG 5: THÔNG TIN HỆ THỐNG
# ============================================================

elif page == "Thông tin hệ thống":
    page_head(
        "Thông tin hệ thống",
        "Kiểm tra trạng thái các mô hình đã huấn luyện và danh sách đặc trưng đầu vào."
    )
    hero(
        "SYSTEM INFORMATION",
        "Thông tin tệp mô hình, trạng thái tải và cấu trúc dữ liệu đầu vào."
    )

    st.subheader("Trạng thái tệp mô hình")
    status_rows = []
    for name, filename in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, filename)
        if name in models:
            status = "Đã tải"
        elif name in model_errors:
            status = "Lỗi khi tải"
        else:
            status = "Không tìm thấy"
        status_rows.append({"Mô hình": name, "Tệp": path, "Trạng thái": status})

    st.dataframe(pd.DataFrame(status_rows), width="stretch", hide_index=True)

    if best_model_name:
        st.write(f"Mô hình được chọn sau huấn luyện: **{best_model_name}**")

    st.subheader(f"Danh sách đặc trưng đầu vào ({len(feature_names)})")
    feature_rows = []
    for group_name, cols in group_features(feature_names).items():
        for feature in cols:
            feature_rows.append({"Nhóm đặc trưng": group_name, "Tên đặc trưng": feature})
    st.dataframe(pd.DataFrame(feature_rows), width="stretch", hide_index=True)

    st.caption(
        "Lưu ý: ứng dụng hiện sử dụng bộ dữ liệu UCI Steel Plates Faults dạng bảng "
        "với các đặc trưng số, không phân loại trực tiếp ảnh chụp tấm thép."
    )
