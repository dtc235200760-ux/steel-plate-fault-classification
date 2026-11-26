
# ============================================================
# 1. KHAI BÁO THƯ VIỆN
# ============================================================

import os
import warnings
import joblib

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)


# ============================================================
# 2. CẤU HÌNH CHƯƠNG TRÌNH
# ============================================================

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

summary_rows = []
trained_models = {}
test_predictions = {}
test_probabilities = {}

print("Đã khởi tạo chương trình!")

# ============================================================
# 3. TẢI VÀ XỬ LÝ DỮ LIỆU UCI
# ============================================================

print("\nĐANG TẢI DỮ LIỆU UCI STEEL PLATES FAULTS...")

try:
    # Thử tải dữ liệu trực tiếp từ UCI
    steel_data = fetch_ucirepo(id=198)

    X = steel_data.data.features.copy()
    y_raw = steel_data.data.targets.copy()

    print("Đã tải dữ liệu thành công từ UCI.")

except Exception as e:

    print("\nKHÔNG THỂ TẢI DỮ LIỆU TỪ UCI.")
    print("Nguyên nhân:", str(e))

    print("\nĐANG CHUYỂN SANG SỬ DỤNG DỮ LIỆU CSV DỰ PHÒNG...")

    # Xác định đường dẫn file CSV trong dự án
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    LOCAL_DATA_FILE = os.path.join(
        BASE_DIR,
        "results",
        "du_lieu_da_xu_ly.csv"
    )

    # Kiểm tra file dữ liệu có tồn tại không
    if not os.path.isfile(LOCAL_DATA_FILE):

        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu dự phòng: {LOCAL_DATA_FILE}"
        )

    # Đọc dữ liệu CSV
    try:
        local_df = pd.read_csv(
            LOCAL_DATA_FILE,
            encoding="utf-8-sig"
        )

    except UnicodeDecodeError:
        local_df = pd.read_csv(
            LOCAL_DATA_FILE,
            encoding="latin1"
        )

    # Chuẩn hóa tên cột
    local_df.columns = (
        local_df.columns.astype(str).str.strip()
    )

    # Loại bỏ cột chỉ mục không cần thiết nếu có
    local_df = local_df.loc[
        :,
        ~local_df.columns.str.startswith("Unnamed:")
    ]

    # Kiểm tra cột nhãn
    if "Loai_loi" not in local_df.columns:

        raise ValueError(
            "File du_lieu_da_xu_ly.csv không có cột Loai_loi."
        )

    # Tách dữ liệu đặc trưng và nhãn
    X = local_df.drop(
        columns=["Loai_loi"]
    ).copy()

    y_raw = local_df[["Loai_loi"]].copy()

    print("Đã đọc dữ liệu dự phòng thành công.")
    print("Nguồn dữ liệu:", LOCAL_DATA_FILE)


# ------------------------------------------------------------
# KIỂM TRA KÍCH THƯỚC DỮ LIỆU
# ------------------------------------------------------------

print("\nKích thước dữ liệu:", X.shape)
print("Kích thước nhãn:", y_raw.shape)


# ------------------------------------------------------------
# KIỂM TRA VÀ XỬ LÝ DỮ LIỆU
# ------------------------------------------------------------

# Kiểm tra dữ liệu thiếu
print("\nSố lượng giá trị thiếu:", X.isnull().sum().sum())

# Kiểm tra dữ liệu vô hạn
X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

# Loại bỏ những dòng có dữ liệu thiếu ở đặc trưng
valid_rows = X.notna().all(axis=1)

X = X.loc[valid_rows].copy()
y_raw = y_raw.loc[valid_rows].copy()


# ------------------------------------------------------------
# CHUYỂN NHÃN ONE-HOT THÀNH NHÃN PHÂN LOẠI
# ------------------------------------------------------------

if y_raw.shape[1] > 1:

    y_raw = y_raw.astype(int)

    if not y_raw.isin([0, 1]).all().all():

        raise ValueError(
            "Các cột nhãn phải chứa giá trị 0 hoặc 1."
        )

    if not (y_raw.sum(axis=1) == 1).all():

        raise ValueError(
            "Mỗi dòng phải có đúng một nhãn lỗi bằng 1."
        )

    y = y_raw.idxmax(axis=1)

else:

    y = y_raw.iloc[:, 0]


# Chuẩn hóa kiểu dữ liệu nhãn
y = y.astype(str).str.strip()

# Chuẩn hóa tên cột đặc trưng
X.columns = X.columns.astype(str).str.strip()

feature_names = list(X.columns)

class_names = sorted(
    y.unique()
)


# ------------------------------------------------------------
# THỐNG KÊ DỮ LIỆU
# ------------------------------------------------------------

print("\nSố lượng mẫu:", len(X))

print(
    "Số lượng đặc trưng:",
    len(feature_names)
)

print(
    "Số loại lỗi:",
    len(class_names)
)

print("\nPHÂN BỐ CÁC LOẠI LỖI:")

print(
    y.value_counts()
)

print("\nHOÀN TẤT TẢI VÀ XỬ LÝ DỮ LIỆU!")

# ============================================================
# 4. PHÂN TÍCH NGOẠI LAI VÀ LƯU DỮ LIỆU
# ============================================================

print("\nĐANG PHÂN TÍCH NGOẠI LAI...")

outlier_rows = []

for column in X.columns:

    Q1 = X[column].quantile(0.25)
    Q3 = X[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    count = (
        (X[column] < lower) |
        (X[column] > upper)
    ).sum()

    outlier_rows.append({
        "Dac_trung": column,
        "So_luong_ngoai_lai": count
    })

outlier_df = pd.DataFrame(outlier_rows)

outlier_df.to_csv(
    "results/phan_tich_ngoai_lai.csv",
    index=False,
    encoding="utf-8-sig"
)

# Không tự động loại bỏ ngoại lai
print("Đã lưu kết quả phân tích ngoại lai.")

# Lưu dữ liệu đã xử lý
processed_data = X.copy()
processed_data["Loai_loi"] = y.values

processed_data.to_csv(
    "results/du_lieu_da_xu_ly.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 5. TRỰC QUAN HÓA PHÂN BỐ LỚP
# ============================================================

plt.figure(figsize=(10, 6))

sns.countplot(
    x=y,
    order=y.value_counts().index
)

plt.title("Phân bố các loại lỗi bề mặt thép")
plt.xlabel("Loại lỗi")
plt.ylabel("Số lượng mẫu")
plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    "results/phan_bo_loai_loi.png",
    dpi=300
)

plt.close()


# ============================================================
# 6. CHIA DỮ LIỆU TRAIN / VALIDATION / TEST
# ============================================================

print("\nĐANG CHIA DỮ LIỆU...")

# 70% train, 30% còn lại
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y
)

# Chia 30% thành 15% validation và 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp
)

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# ============================================================
# 7. KHAI BÁO 5 MÔ HÌNH HỌC MÁY
# ============================================================

model_configs = {

    "SVM": {
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                probability=True,
                random_state=RANDOM_STATE
            ))
        ]),

        "params": {
            "classifier__C": [1, 5, 10],
            "classifier__gamma": ["scale", "auto"],
            "classifier__kernel": ["rbf"]
        }
    },

    "KNN": {
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier())
        ]),

        "params": {
            "classifier__n_neighbors": [3, 5, 7],
            "classifier__weights": ["uniform", "distance"],
            "classifier__metric": ["euclidean", "manhattan"]
        }
    },

    "Random Forest": {
        "model": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "params": {
            "n_estimators": [100, 200],
            "max_depth": [None, 20],
            "min_samples_split": [2, 5]
        }
    },

    "Extra Trees": {
        "model": ExtraTreesClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "params": {
            "n_estimators": [100, 200],
            "max_depth": [None, 20],
            "min_samples_split": [2, 5]
        }
    },

    "Gradient Boosting": {
        "model": GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),

        "params": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [2, 3]
        }
    }
}


# ============================================================
# 8. HÀM ĐÁNH GIÁ MÔ HÌNH
# ============================================================

def evaluate_model(model, X_data, y_true):

    y_pred = model.predict(X_data)

    y_prob = model.predict_proba(X_data)

    labels = model.classes_

    y_true_binary = label_binarize(
        y_true,
        classes=labels
    )

    try:
        auc_macro = roc_auc_score(
            y_true_binary,
            y_prob,
            average="macro",
            multi_class="ovr"
        )
    except ValueError:
        auc_macro = np.nan

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),

        "Precision_Macro": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "Recall_Macro": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "F1_Macro": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "AUC_Macro_OVR": auc_macro
    }

    return y_pred, y_prob, metrics


# ============================================================
# 9. HUẤN LUYỆN VÀ TỐI ƯU 5 MÔ HÌNH
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

for name, config in model_configs.items():

    print("\n" + "=" * 60)
    print("ĐANG HUẤN LUYỆN:", name)
    print("=" * 60)

    grid = GridSearchCV(
        estimator=config["model"],
        param_grid=config["params"],
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    print("Tham số tốt nhất:", grid.best_params_)
    print("CV F1 Macro:", grid.best_score_)

    # Đánh giá trên 3 tập
    train_pred, train_prob, train_metrics = evaluate_model(
        best_model,
        X_train,
        y_train
    )

    val_pred, val_prob, val_metrics = evaluate_model(
        best_model,
        X_val,
        y_val
    )

    test_pred, test_prob, test_metrics = evaluate_model(
        best_model,
        X_test,
        y_test
    )

    # Lưu kết quả tổng hợp
    row = {
        "Mo_hinh": name,
        "Best_Params": str(grid.best_params_),
        "CV_F1_Macro": grid.best_score_
    }

    for metric, value in train_metrics.items():
        row["Train_" + metric] = value

    for metric, value in val_metrics.items():
        row["Val_" + metric] = value

    for metric, value in test_metrics.items():
        row["Test_" + metric] = value

    summary_rows.append(row)

    # Lưu mô hình đã huấn luyện
    trained_models[name] = best_model

    test_predictions[name] = test_pred
    test_probabilities[name] = test_prob

    safe_name = name.lower().replace(" ", "_")

    joblib.dump(
        best_model,
        f"models/{safe_name}.joblib"
    )

    print("\nKẾT QUẢ TEST:")
    print(classification_report(
        y_test,
        test_pred,
        zero_division=0
    ))

    # Ma trận nhầm lẫn
    cm = confusion_matrix(
        y_test,
        test_pred,
        labels=class_names
    )

    plt.figure(figsize=(9, 7))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title("Confusion Matrix - " + name)
    plt.xlabel("Nhãn dự đoán")
    plt.ylabel("Nhãn thực tế")
    plt.tight_layout()

    plt.savefig(
        f"results/confusion_matrix_{safe_name}.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 10. SO SÁNH KẾT QUẢ CÁC MÔ HÌNH
# ============================================================

print("\nĐANG TỔNG HỢP KẾT QUẢ...")

results_df = pd.DataFrame(summary_rows)

# Sắp xếp theo F1 Macro trên validation
results_df = results_df.sort_values(
    by="Val_F1_Macro",
    ascending=False
)

results_df.to_csv(
    "results/so_sanh_mo_hinh.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nBẢNG SO SÁNH CÁC MÔ HÌNH:")
print(results_df.to_string(index=False))


# ============================================================
# 11. LƯU MÔ HÌNH TỐT NHẤT
# ============================================================

best_name = results_df.iloc[0]["Mo_hinh"]

best_model = trained_models[best_name]

joblib.dump(
    best_model,
    "models/best_model.joblib"
)

joblib.dump(
    best_name,
    "models/best_model_name.joblib"
)

joblib.dump(
    feature_names,
    "models/feature_names.joblib"
)

print("\nMÔ HÌNH TỐT NHẤT:", best_name)

# ============================================================
# 11.1. VẼ ROC CURVE OVR CHO CẢ 5 MÔ HÌNH
# ============================================================

print("\nĐANG VẼ ROC CURVE CHO 5 MÔ HÌNH...")

ROC_RESULT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results"
)

os.makedirs(ROC_RESULT_DIR, exist_ok=True)

for model_name, model in trained_models.items():

    print(f"\nĐANG VẼ ROC: {model_name}")

    # Lấy xác suất dự đoán trên tập Test
    y_prob = np.asarray(
        test_probabilities[model_name]
    )

    # Lấy thứ tự các lớp mà mô hình đã học
    model_classes = np.asarray(
        model.classes_
    )

    # Chuyển nhãn thực tế sang One-vs-Rest
    y_test_binary = label_binarize(
        y_test,
        classes=model_classes
    )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    plotted_classes = 0

    # Vẽ ROC cho từng loại lỗi
    for i, class_name in enumerate(model_classes):

        binary_labels = y_test_binary[:, i]

        # Kiểm tra phải có cả mẫu dương và mẫu âm
        if np.unique(binary_labels).size < 2:
            continue

        fpr, tpr, _ = roc_curve(
            binary_labels,
            y_prob[:, i]
        )

        roc_auc_value = auc(
            fpr,
            tpr
        )

        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            label=(
                f"{class_name} "
                f"(AUC = {roc_auc_value:.3f})"
            )
        )

        plotted_classes += 1

    # Đường phân loại ngẫu nhiên
    ax.plot(
        [0, 1],
        [0, 1],
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Ngẫu nhiên"
    )

    ax.set_title(
        f"Đường cong ROC AUC - {model_name}",
        fontsize=16,
        fontweight="bold"
    )

    ax.set_xlabel(
        "False Positive Rate",
        fontsize=12
    )

    ax.set_ylabel(
        "True Positive Rate",
        fontsize=12
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    ax.legend(
        loc="lower right",
        fontsize=9
    )

    ax.grid(
        alpha=0.2
    )

    plt.tight_layout()

    # Tên file theo từng mô hình
    safe_name = (
        model_name.lower().replace(" ", "_")
    )

    output_path = os.path.join(
        ROC_RESULT_DIR,
        f"roc_{safe_name}.png"
    )

    if plotted_classes > 0:

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        print("Đã lưu:", output_path)

        # Lưu thêm ảnh ROC của mô hình tốt nhất
        if model_name == best_name:

            best_roc_path = os.path.join(
                ROC_RESULT_DIR,
                "roc_curve_best_model.png"
            )

            plt.savefig(
                best_roc_path,
                dpi=300,
                bbox_inches="tight"
            )

            print(
                "Đã lưu ROC mô hình tốt nhất:",
                best_roc_path
            )

    else:

        print(
            f"Không đủ dữ liệu để vẽ ROC cho {model_name}"
        )

    plt.close(fig)

print("\nĐÃ HOÀN THÀNH VẼ ROC CHO 5 MÔ HÌNH!")


# ============================================================
# 12. VẼ 5 BIỂU ĐỒ SO SÁNH MÔ HÌNH TRÊN TẬP TEST
# MỖI MÔ HÌNH MỘT MÀU - HIỂN THỊ PHẦN TRĂM TRONG CỘT
# ============================================================
# ------------------------------------------------------------
# 1. TẠO THƯ MỤC RESULTS
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(RESULT_DIR, exist_ok=True)


# ------------------------------------------------------------
# 2. KHAI BÁO 5 CHỈ SỐ ĐÁNH GIÁ
# ------------------------------------------------------------

metrics_to_plot = [
    ("Test_Accuracy", "Accuracy", "so_sanh_test_accuracy.png"),

    ("Test_Precision_Macro", "Precision", "so_sanh_test_precision_macro.png"),

    ("Test_Recall_Macro", "Recall", "so_sanh_test_recall_macro.png"),

    ("Test_F1_Macro", "F1-score", "so_sanh_test_f1_macro.png"),

    ("Test_AUC_Macro_OVR", "ROC-AUC", "so_sanh_test_auc_macro_ovr.png")
]


# ------------------------------------------------------------
# 3. QUY ĐỊNH MÀU RIÊNG CHO TỪNG MÔ HÌNH
# ------------------------------------------------------------

model_colors = {
    "SVM": "#4285F4",
    "KNN": "#14B8A6",
    "Random Forest": "#F59E0B",
    "Extra Trees": "#8B5CF6",
    "Gradient Boosting": "#EF4444"
}

# Thứ tự hiển thị cố định
model_order = [
    "SVM",
    "KNN",
    "Random Forest",
    "Extra Trees",
    "Gradient Boosting"
]

# Chuẩn hóa tên mô hình để tránh sai khác cách viết
def normalize_model_name(name):

    key = (
        str(name)
        .replace("_", " ")
        .replace("-", " ")
        .strip()
        .lower()
    )

    key = " ".join(key.split())

    aliases = {
        "svm": "SVM",
        "knn": "KNN",
        "rf": "Random Forest",
        "randomforest": "Random Forest",
        "random forest": "Random Forest",
        "et": "Extra Trees",
        "extratrees": "Extra Trees",
        "extra trees": "Extra Trees",
        "gb": "Gradient Boosting",
        "gradientboosting": "Gradient Boosting",
        "gradient boosting": "Gradient Boosting"
    }

    return aliases.get(key, str(name).strip())


# ------------------------------------------------------------
# 4. KIỂM TRA DỮ LIỆU
# ------------------------------------------------------------

if "Mo_hinh" not in results_df.columns:

    raise ValueError(
        "Không tìm thấy cột Mo_hinh trong results_df."
    )

# Chuẩn hóa tên mô hình
plot_source = results_df.copy()

plot_source["Mo_hinh"] = (
    plot_source["Mo_hinh"]
    .apply(normalize_model_name)
)

# Kiểm tra đủ 5 mô hình
missing_models = [
    name for name in model_order
    if name not in plot_source["Mo_hinh"].values
]

if missing_models:

    raise ValueError(
        f"Thiếu kết quả của các mô hình: {missing_models}"
    )


# ------------------------------------------------------------
# 5. VẼ TỪNG BIỂU ĐỒ
# ------------------------------------------------------------

print("\nĐANG VẼ 5 BIỂU ĐỒ SO SÁNH...")

for metric, title, filename in metrics_to_plot:

    # Kiểm tra cột chỉ số
    if metric not in plot_source.columns:

        raise ValueError(
            f"Không tìm thấy cột {metric} trong results_df."
        )

    # Lấy dữ liệu của chỉ số hiện tại
    plot_df = plot_source[
        ["Mo_hinh", metric]
    ].copy()

    plot_df[metric] = pd.to_numeric(
        plot_df[metric],
        errors="coerce"
    )

    plot_df = plot_df.dropna(
        subset=[metric]
    )

    # Sắp xếp theo thứ tự mô hình cố định
    plot_df["Mo_hinh"] = pd.Categorical(
        plot_df["Mo_hinh"],
        categories=model_order,
        ordered=True
    )

    plot_df = plot_df.sort_values(
        "Mo_hinh"
    )

    if len(plot_df) != 5:

        raise ValueError(
            f"Chỉ số {metric} không có đủ kết quả của 5 mô hình."
        )

    # Lấy tên đầy đủ
    labels = plot_df["Mo_hinh"].astype(str).tolist()

    # Lấy giá trị đánh giá
    values = plot_df[metric].to_numpy(dtype=float)

    if not np.isfinite(values).all():

        raise ValueError(
            f"Dữ liệu {metric} chứa giá trị không hợp lệ."
        )

    # Chuyển giá trị sang phần trăm
    # Hỗ trợ cả dữ liệu dạng 0-1 và 0-100
    if values.max() <= 1.0:

        percentages = values * 100

    else:

        percentages = values.copy()

    # Lấy màu tương ứng với từng mô hình
    bar_colors = [
        model_colors[name]
        for name in labels
    ]

    # --------------------------------------------------------
    # VẼ BIỂU ĐỒ
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    bars = ax.bar(
        labels,
        percentages,
        color=bar_colors,
        width=0.68,
        edgecolor="none"
    )

    # Tiêu đề
    ax.set_title(
        title,
        fontsize=20,
        fontweight="bold",
        loc="center",
        pad=20
    )

    ax.set_xlabel("")
    ax.set_ylabel("")

    # Trục tung từ 0% đến 100%
    ax.set_ylim(0, 100)

    ax.set_yticks([
        0, 25, 50, 75, 100
    ])

    ax.set_yticklabels([
        "0%", "25%", "50%", "75%", "100%"
    ])

    # Hiển thị phần trăm bên trong, gần đầu mỗi cột
    for bar, value in zip(bars, percentages):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 0.90,
            f"{value:.2f}%",
            ha="center",
            va="center",
            color="white",
            fontsize=11,
            fontweight="bold"
        )

    # Hiển thị đầy đủ tên mô hình
    ax.set_xticks(range(len(labels)))

    ax.set_xticklabels(
        labels,
        rotation=0,
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold"
    )

    ax.tick_params(
        axis="x",
        length=0,
        pad=12
    )

    # Đường lưới ngang
    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.20
    )

    ax.set_axisbelow(True)

    # Xóa đường viền
    for spine in ax.spines.values():

        spine.set_visible(False)

    # Khoảng cách biểu đồ
    plt.tight_layout()

    # Lưu ảnh
    output_path = os.path.join(
        RESULT_DIR,
        filename
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

    print(f"Đã lưu biểu đồ {title}: {output_path}")


print("\n" + "=" * 60)
print("ĐÃ VẼ THÀNH CÔNG 5 BIỂU ĐỒ SO SÁNH!")
print("=" * 60)

# ============================================================
# 12.1. VẼ 5 BIỂU ĐỒ ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TRAIN
# ============================================================

print("\nĐANG VẼ 5 BIỂU ĐỒ TRÊN TẬP TRAIN...")

train_metrics_to_plot = [
    ("Train_Accuracy", "Accuracy", "so_sanh_train_accuracy.png"),

    ("Train_Precision_Macro", "Precision", "so_sanh_train_precision_macro.png"),

    ("Train_Recall_Macro", "Recall", "so_sanh_train_recall_macro.png"),

    ("Train_F1_Macro", "F1-score", "so_sanh_train_f1_macro.png"),

    ("Train_AUC_Macro_OVR", "ROC-AUC", "so_sanh_train_auc_macro_ovr.png")
]

for metric, title, filename in train_metrics_to_plot:

    # Kiểm tra cột dữ liệu
    if metric not in plot_source.columns:

        raise ValueError(
            f"Không tìm thấy cột {metric} trong results_df."
        )

    # Lấy dữ liệu tập Train
    train_df = plot_source[
        ["Mo_hinh", metric]
    ].copy()

    train_df[metric] = pd.to_numeric(
        train_df[metric],
        errors="coerce"
    )

    train_df = train_df.dropna(
        subset=[metric]
    )

    # Sắp xếp theo thứ tự mô hình
    train_df["Mo_hinh"] = pd.Categorical(
        train_df["Mo_hinh"],
        categories=model_order,
        ordered=True
    )

    train_df = train_df.sort_values(
        "Mo_hinh"
    )

    if len(train_df) != 5:

        raise ValueError(
            f"Không đủ dữ liệu của 5 mô hình cho {metric}."
        )

    labels = train_df["Mo_hinh"].astype(str).tolist()

    values = train_df[metric].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            f"Dữ liệu {metric} không hợp lệ."
        )

    # Chuyển về phần trăm
    percentages = (
        values * 100
        if values.max() <= 1
        else values
    )

    # Mỗi mô hình một màu
    bar_colors = [
        model_colors[name]
        for name in labels
    ]

    # Tạo biểu đồ
    fig, ax = plt.subplots(
        figsize=(13, 7)
    )

    bars = ax.bar(
        labels,
        percentages,
        color=bar_colors,
        width=0.68,
        edgecolor="none"
    )

    # Tiêu đề căn giữa
    ax.set_title(
        f"{title} - TẬP TRAIN",
        fontsize=20,
        fontweight="bold",
        loc="center",
        pad=20
    )

    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.set_ylim(0, 100)

    ax.set_yticks([
        0, 25, 50, 75, 100
    ])

    ax.set_yticklabels([
        "0%", "25%", "50%", "75%", "100%"
    ])

    # Hiển thị phần trăm bên trong mỗi cột
    for bar, value in zip(bars, percentages):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 0.90,
            f"{value:.2f}%",
            ha="center",
            va="center",
            color="white",
            fontsize=11,
            fontweight="bold"
        )

    # Tên mô hình căn giữa
    ax.set_xticks(range(len(labels)))

    ax.set_xticklabels(
        labels,
        rotation=0,
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold"
    )

    ax.tick_params(
        axis="x",
        length=0,
        pad=12
    )

    # Đường lưới
    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.20
    )

    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    # Lưu ảnh vào results
    output_path = os.path.join(
        RESULT_DIR,
        filename
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

    print(f"Đã lưu biểu đồ Train: {output_path}")

print("\nĐÃ VẼ XONG 5 BIỂU ĐỒ TRAIN!")

# ============================================================
# 12.2. VẼ 5 BIỂU ĐỒ ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP VALIDATION
# ============================================================

print("\nĐANG VẼ 5 BIỂU ĐỒ TRÊN TẬP VALIDATION...")

validation_metrics_to_plot = [
    (
        "Val_Accuracy",
        "Accuracy",
        "so_sanh_val_accuracy.png"
    ),
    (
        "Val_Precision_Macro",
        "Precision",
        "so_sanh_val_precision_macro.png"
    ),
    (
        "Val_Recall_Macro",
        "Recall",
        "so_sanh_val_recall_macro.png"
    ),
    (
        "Val_F1_Macro",
        "F1-score",
        "so_sanh_val_f1_macro.png"
    ),
    (
        "Val_AUC_Macro_OVR",
        "ROC-AUC",
        "so_sanh_val_auc_macro_ovr.png"
    )
]

# Duyệt qua từng chỉ số đánh giá
for metric, title, filename in validation_metrics_to_plot:

    # Kiểm tra cột dữ liệu
    if metric not in plot_source.columns:

        raise ValueError(
            f"Không tìm thấy cột {metric} trong results_df."
        )

    # Lấy dữ liệu Validation
    val_df = plot_source[
        ["Mo_hinh", metric]
    ].copy()

    val_df[metric] = pd.to_numeric(
        val_df[metric],
        errors="coerce"
    )

    val_df = val_df.dropna(
        subset=[metric]
    )

    # Sắp xếp theo thứ tự mô hình
    val_df["Mo_hinh"] = pd.Categorical(
        val_df["Mo_hinh"],
        categories=model_order,
        ordered=True
    )

    val_df = val_df.sort_values(
        "Mo_hinh"
    )

    if len(val_df) != 5:

        raise ValueError(
            f"Không đủ dữ liệu của 5 mô hình cho {metric}."
        )

    # Lấy tên mô hình đầy đủ
    labels = val_df["Mo_hinh"].astype(str).tolist()

    # Lấy giá trị đánh giá
    values = val_df[metric].to_numpy(dtype=float)

    if not np.isfinite(values).all():

        raise ValueError(
            f"Dữ liệu {metric} không hợp lệ."
        )

    # Chuyển sang phần trăm
    percentages = (
        values * 100
        if values.max() <= 1.0
        else values
    )

    # Màu riêng cho từng mô hình
    bar_colors = [
        model_colors[name]
        for name in labels
    ]

    # --------------------------------------------------------
    # VẼ BIỂU ĐỒ
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(13, 7)
    )

    bars = ax.bar(
        labels,
        percentages,
        color=bar_colors,
        width=0.68,
        edgecolor="none"
    )

    # Tiêu đề căn giữa
    ax.set_title(
        f"{title} - TẬP VALIDATION",
        fontsize=20,
        fontweight="bold",
        loc="center",
        pad=20
    )

    ax.set_xlabel("")
    ax.set_ylabel("")

    # Trục tung từ 0% đến 100%
    ax.set_ylim(0, 100)

    ax.set_yticks([
        0, 25, 50, 75, 100
    ])

    ax.set_yticklabels([
        "0%", "25%", "50%", "75%", "100%"
    ])

    # Phần trăm bên trong mỗi cột
    for bar, value in zip(bars, percentages):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 0.90,
            f"{value:.2f}%",
            ha="center",
            va="center",
            color="white",
            fontsize=11,
            fontweight="bold"
        )

    # Tên mô hình căn giữa dưới từng cột
    ax.set_xticks(range(len(labels)))

    ax.set_xticklabels(
        labels,
        rotation=0,
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold"
    )

    ax.tick_params(
        axis="x",
        length=0,
        pad=12
    )

    # Đường lưới ngang
    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.20
    )

    ax.set_axisbelow(True)

    # Xóa đường viền
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    # Lưu ảnh vào results
    output_path = os.path.join(
        RESULT_DIR,
        filename
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

    print(
        f"Đã lưu biểu đồ Validation: {output_path}"
    )

print("\nĐÃ VẼ XONG 5 BIỂU ĐỒ VALIDATION!")

# ============================================================
# 13. TỰ ĐỘNG PHÂN TÍCH VÀ NHẬN XÉT KẾT QUẢ
# ============================================================

print("\nĐANG TẠO NHẬN XÉT KẾT QUẢ...")

metric_names = {
    "Test_Accuracy": "Accuracy",
    "Test_Precision_Macro": "Precision",
    "Test_Recall_Macro": "Recall",
    "Test_F1_Macro": "F1-score",
    "Test_AUC_Macro_OVR": "ROC-AUC"
}

analysis_df = results_df.copy()

for metric in metric_names:

    analysis_df[metric] = pd.to_numeric(
        analysis_df[metric],
        errors="coerce"
    )

analysis_df = analysis_df.dropna(
    subset=list(metric_names.keys())
)

if analysis_df.empty:
    raise ValueError(
        "Không có dữ liệu hợp lệ để tạo nhận xét."
    )


def get_metric_ranking(metric):

    return analysis_df.sort_values(
        metric,
        ascending=False
    ).reset_index(drop=True)


def get_metric_top(metric):

    max_value = analysis_df[metric].max()

    best_models = analysis_df[
        np.isclose(
            analysis_df[metric],
            max_value,
            atol=1e-8
        )
    ]["Mo_hinh"].tolist()

    return best_models, max_value


def format_model_score(name, value, percentage=True):

    if percentage:
        return f"{name} đạt {value * 100:.2f}%"

    return f"{name} đạt {value:.4f}"


def get_remaining_ranking(metric, best_models, percentage=True):

    ranking = get_metric_ranking(metric)

    ranking = ranking[
        ~ranking["Mo_hinh"].isin(best_models)
    ]

    parts = []

    for _, row in ranking.iterrows():

        parts.append(
            format_model_score(
                row["Mo_hinh"],
                row[metric],
                percentage
            )
        )

    return "; ".join(parts)


# Số mẫu kiểm thử
n_test = len(y_test)

report_sections = []


# ------------------------------------------------------------
# 1. ACCURACY
# ------------------------------------------------------------

best_models, best_value = get_metric_top(
    "Test_Accuracy"
)

ranking = get_metric_ranking(
    "Test_Accuracy"
)

best_names = " và ".join(best_models)

if len(best_models) > 1:

    accuracy_text = (
        f"{best_names} cùng đạt Accuracy cao nhất, "
        f"ở mức {best_value:.4f} ({best_value * 100:.2f}%). "
        f"Điều này cho thấy các mô hình có tỷ lệ dự đoán đúng "
        f"tổng thể tương đương nhau trên {n_test} mẫu kiểm thử."
    )

else:

    accuracy_text = (
        f"{best_names} đạt Accuracy cao nhất, "
        f"ở mức {best_value:.4f} ({best_value * 100:.2f}%). "
        f"Đây là mô hình có tỷ lệ dự đoán đúng tổng thể cao nhất "
        f"trên {n_test} mẫu kiểm thử."
    )

remaining = get_remaining_ranking(
    "Test_Accuracy",
    best_models
)

if remaining:
    accuracy_text += (
        f"\n\nCác mô hình còn lại lần lượt đạt: {remaining}."
    )

lowest_accuracy = ranking.iloc[-1]

accuracy_text += (
    f" Như vậy, {lowest_accuracy['Mo_hinh']} "
    "có Accuracy thấp nhất trong năm mô hình."
)

report_sections.append(
    ("Accuracy", accuracy_text)
)


# ------------------------------------------------------------
# 2. PRECISION
# ------------------------------------------------------------

best_models, best_value = get_metric_top(
    "Test_Precision_Macro"
)

precision_text = (
    f"{' và '.join(best_models)} "
    f"{'cùng đạt' if len(best_models) > 1 else 'đạt'} "
    f"Precision cao nhất với {best_value:.4f} "
    f"({best_value * 100:.2f}%)."
)

remaining = get_remaining_ranking(
    "Test_Precision_Macro",
    best_models
)

if remaining:
    precision_text += (
        f" Các mô hình tiếp theo: {remaining}."
    )

precision_text += (
    "\n\nPrecision phản ánh mức độ chính xác của các dự đoán "
    "thuộc từng lớp. Với Precision Macro, kết quả được tính "
    "bằng cách lấy trung bình Precision của các lớp, "
    "do đó mỗi lớp có vai trò ngang nhau trong chỉ số tổng hợp."
)

report_sections.append(
    ("Precision", precision_text)
)


# ------------------------------------------------------------
# 3. RECALL
# ------------------------------------------------------------

best_models, best_value = get_metric_top(
    "Test_Recall_Macro"
)

recall_text = (
    f"{' và '.join(best_models)} "
    f"{'cùng đạt' if len(best_models) > 1 else 'đạt'} "
    f"Recall cao nhất với {best_value:.4f} "
    f"({best_value * 100:.2f}%)."
)

remaining = get_remaining_ranking(
    "Test_Recall_Macro",
    best_models
)

if remaining:
    recall_text += (
        f" Các mô hình tiếp theo: {remaining}."
    )

recall_text += (
    "\n\nRecall là chỉ số quan trọng trong bài toán phát hiện lỗi "
    "vì phản ánh khả năng nhận diện các mẫu thực sự thuộc một lớp lỗi. "
    "Recall Macro giúp đánh giá mức độ nhận diện trên cả bảy loại lỗi "
    "mà không để lớp có nhiều mẫu chi phối hoàn toàn kết quả."
)

report_sections.append(
    ("Recall", recall_text)
)


# ------------------------------------------------------------
# 4. F1-SCORE
# ------------------------------------------------------------

best_models, best_value = get_metric_top(
    "Test_F1_Macro"
)

f1_text = (
    f"{' và '.join(best_models)} "
    f"{'cùng đạt' if len(best_models) > 1 else 'đạt'} "
    f"F1-score cao nhất với {best_value:.4f} "
    f"({best_value * 100:.2f}%)."
)

remaining = get_remaining_ranking(
    "Test_F1_Macro",
    best_models
)

if remaining:
    f1_text += (
        f" Các mô hình tiếp theo: {remaining}."
    )

f1_text += (
    "\n\nF1-score là trung bình điều hòa của Precision và Recall. "
    "F1 Macro được tính bằng cách lấy trung bình F1-score của "
    "các lớp. Vì vậy, mô hình có F1 Macro cao nhất thể hiện "
    "sự cân bằng tốt nhất giữa Precision và Recall "
    "khi xét đồng đều các lớp."
)

report_sections.append(
    ("F1-score", f1_text)
)


# ------------------------------------------------------------
# 5. ROC-AUC
# ------------------------------------------------------------

best_models, best_value = get_metric_top(
    "Test_AUC_Macro_OVR"
)

auc_text = (
    f"{' và '.join(best_models)} "
    f"{'cùng đạt' if len(best_models) > 1 else 'đạt'} "
    f"ROC-AUC cao nhất với {best_value:.4f}."
)

remaining = get_remaining_ranking(
    "Test_AUC_Macro_OVR",
    best_models,
    percentage=False
)

if remaining:
    auc_text += (
        f" Các mô hình tiếp theo: {remaining}."
    )

auc_text += (
    "\n\nROC-AUC Macro OvR đánh giá khả năng phân biệt từng lớp "
    "với các lớp còn lại, sau đó lấy trung bình kết quả. "
    "Giá trị càng gần 1 cho thấy khả năng phân biệt các lớp "
    "càng tốt."
)

report_sections.append(
    ("ROC-AUC", auc_text)
)


# ============================================================
# 14. HIỂN THỊ VÀ LƯU BÁO CÁO PHÂN TÍCH
# ============================================================

full_report = (
    "PHÂN TÍCH VÀ SO SÁNH KẾT QUẢ CÁC MÔ HÌNH\n"
    "ĐÁNH GIÁ TRÊN TẬP TEST\n\n"
)

for title, content in report_sections:

    section_text = (
        f"So sánh theo {title}:\n\n"
        f"{content}\n\n"
    )

    full_report += section_text

    print("\n" + section_text)

# Lưu báo cáo nhận xét
report_path = os.path.join(
    RESULT_DIR,
    "phan_tich_so_sanh_mo_hinh.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(full_report)

print("\nĐÃ LƯU BÁO CÁO:")
print(report_path)

print("\nHOÀN TẤT VẼ BIỂU ĐỒ VÀ PHÂN TÍCH!")

# ============================================================
# 13. TỰ ĐỘNG PHÂN TÍCH VÀ NHẬN XÉT KẾT QUẢ
# ============================================================

print("\n")
print("=" * 75)
print("PHÂN TÍCH VÀ SO SÁNH KẾT QUẢ CÁC MÔ HÌNH")
print("=" * 75)

# Các chỉ số cần phân tích
analysis_metrics = {
    "Test_Accuracy": "Accuracy",
    "Test_Precision_Macro": "Precision",
    "Test_Recall_Macro": "Recall",
    "Test_F1_Macro": "F1-score",
    "Test_AUC_Macro_OVR": "ROC-AUC"
}

# Kiểm tra dữ liệu
for metric in analysis_metrics:
    if metric not in results_df.columns:
        raise ValueError(
            f"Không tìm thấy cột {metric} trong results_df."
        )

# Chuyển các cột chỉ số về dạng số
analysis_df = results_df.copy()

for metric in analysis_metrics:
    analysis_df[metric] = pd.to_numeric(
        analysis_df[metric],
        errors="coerce"
    )

analysis_df = analysis_df.dropna(
    subset=list(analysis_metrics.keys())
)

if analysis_df.empty:
    raise ValueError(
        "Không có dữ liệu hợp lệ để phân tích."
    )


# Hàm định dạng kết quả
def format_metric(value, show_percent=True):

    if show_percent:
        return f"{value:.4f} ({value * 100:.2f}%)"

    return f"{value:.4f}"


# Hàm lấy tên mô hình tốt nhất, hỗ trợ trường hợp đồng hạng
def get_best_models(metric):

    max_value = analysis_df[metric].max()

    best_models = analysis_df[
        np.isclose(
            analysis_df[metric],
            max_value,
            atol=1e-8
        )
    ]["Mo_hinh"].tolist()

    return best_models, max_value


# Hàm tạo danh sách xếp hạng
def ranking_models(metric, show_percent=True):

    sorted_df = analysis_df.sort_values(
        by=metric,
        ascending=False
    )

    ranking = []

    for _, row in sorted_df.iterrows():

        model_name = row["Mo_hinh"]
        value = row[metric]

        ranking.append(
            f"{model_name}: "
            f"{format_metric(value, show_percent)}"
        )

    return ", ".join(ranking)


# Danh sách nhận xét
analysis_text = []

# ------------------------------------------------------------
# 1. ACCURACY
# ------------------------------------------------------------

best_models, best_value = get_best_models(
    "Test_Accuracy"
)

best_names = " và ".join(best_models)

remaining = analysis_df[
    ~analysis_df["Mo_hinh"].isin(best_models)
].sort_values(
    "Test_Accuracy",
    ascending=False
)

n_test = len(y_test)

accuracy_text = (
    "**So sánh theo Accuracy:**\n\n"
    f"{best_names} "
    f"{'cùng đạt' if len(best_models) > 1 else 'đạt'} "
    f"Accuracy cao nhất, ở mức "
    f"{format_metric(best_value)}. "
    f"Điều này cho thấy "
)

if len(best_models) > 1:
    accuracy_text += (
        "hai mô hình có tỷ lệ dự đoán đúng tổng thể tương đương nhau "
    )
else:
    accuracy_text += (
        "mô hình có tỷ lệ dự đoán đúng tổng thể cao nhất "
    )

accuracy_text += (
    f"trên {n_test} mẫu kiểm thử.\n\n"
)

if not remaining.empty:

    remaining_text = ranking_models(
        "Test_Accuracy"
    )

    accuracy_text += (
        f"Các mô hình được xếp theo Accuracy giảm dần: "
        f"{remaining_text}. "
        f"{remaining.iloc[-1]['Mo_hinh']} "
        "có Accuracy thấp nhất trong các mô hình được thử nghiệm."
    )

analysis_text.append(accuracy_text)


# ------------------------------------------------------------
# 2. PRECISION
# ------------------------------------------------------------

best_models, best_value = get_best_models(
    "Test_Precision_Macro"
)

precision_text = (
    "**So sánh theo Precision:**\n\n"
    f"{' và '.join(best_models)} "
    f"{'cùng đạt' if len(best_models) > 1 else 'đạt'} "
    f"Precision cao nhất với giá trị "
    f"{format_metric(best_value)}. "
    f"Thứ tự kết quả các mô hình: "
    f"{ranking_models('Test_Precision_Macro')}.\n\n"
    "Kết quả cho thấy mô hình đứng đầu có lợi thế về Precision, "
    "tức là có khả năng hạn chế các trường hợp dự đoán một mẫu "
    "thuộc lớp lỗi khi mẫu đó thực tế không thuộc lớp này."
)

analysis_text.append(precision_text)


# ------------------------------------------------------------
# 3. RECALL
# ------------------------------------------------------------

best_models, best_value = get_best_models(
    "Test_Recall_Macro"
)

recall_text = (
    "**So sánh theo Recall:**\n\n"
    f"{' và '.join(best_models)} "
    f"{'cùng đạt' if len(best_models) > 1 else 'đạt'} "
    f"Recall cao nhất với giá trị "
    f"{format_metric(best_value)}. "
    f"Thứ tự kết quả các mô hình: "
    f"{ranking_models('Test_Recall_Macro')}.\n\n"
    "Recall là chỉ số quan trọng trong bài toán phát hiện lỗi "
    "vì phản ánh khả năng nhận diện các mẫu thực sự thuộc một lớp lỗi. "
    "Do đó, mô hình đứng đầu có kết quả tốt nhất theo chỉ số Recall "
    "trên tập kiểm thử."
)

analysis_text.append(recall_text)


# ------------------------------------------------------------
# 4. F1-SCORE
# ------------------------------------------------------------

best_models, best_value = get_best_models(
    "Test_F1_Macro"
)

f1_text = (
    "**So sánh theo F1-score:**\n\n"
    f"{' và '.join(best_models)} "
    f"{'cùng đạt' if len(best_models) > 1 else 'đạt'} "
    f"F1-score cao nhất với giá trị "
    f"{format_metric(best_value)}. "
    f"Thứ tự kết quả các mô hình: "
    f"{ranking_models('Test_F1_Macro')}.\n\n"
    "F1-score thể hiện sự cân bằng giữa Precision và Recall "
    "trên các lớp. Vì vậy, mô hình đứng đầu theo F1 Macro "
    "có kết quả cân bằng tốt nhất giữa hai chỉ số này."
)

analysis_text.append(f1_text)


# ------------------------------------------------------------
# 5. ROC-AUC
# ------------------------------------------------------------

best_models, best_value = get_best_models(
    "Test_AUC_Macro_OVR"
)

auc_text = (
    "**So sánh theo ROC-AUC:**\n\n"
    f"{' và '.join(best_models)} "
    f"{'cùng đạt' if len(best_models) > 1 else 'đạt'} "
    f"ROC-AUC cao nhất với giá trị "
    f"{format_metric(best_value, show_percent=False)}. "
    f"Thứ tự kết quả các mô hình: "
    f"{ranking_models('Test_AUC_Macro_OVR', show_percent=False)}.\n\n"
    "Các giá trị ROC-AUC cho thấy khả năng phân biệt các lớp "
    "của các mô hình trên tập kiểm thử. Giá trị càng gần 1 "
    "thì khả năng phân biệt giữa các lớp càng tốt."
)

analysis_text.append(auc_text)


# ------------------------------------------------------------
# HIỂN THỊ VÀ LƯU KẾT QUẢ PHÂN TÍCH
# ------------------------------------------------------------

full_analysis = "\n\n".join(analysis_text)

print("\n" + full_analysis)

os.makedirs("results", exist_ok=True)

report_path = os.path.join(
    "results",
    "phan_tich_so_sanh_mo_hinh.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "PHÂN TÍCH VÀ SO SÁNH KẾT QUẢ CÁC MÔ HÌNH\n\n"
    )

    file.write(full_analysis)

print("\nĐã lưu nhận xét tại:")
print(report_path)

print("\nHOÀN TẤT PHÂN TÍCH KẾT QUẢ!")


# ============================================================
# 14. PHÂN TÍCH CÁC LỚP LỖI KHÓ PHÂN BIỆT
# ============================================================

best_pred = test_predictions[best_name]

cm = confusion_matrix(
    y_test,
    best_pred,
    labels=class_names
)

confusion_details = []

for i, actual_class in enumerate(class_names):

    for j, predicted_class in enumerate(class_names):

        if i != j and cm[i, j] > 0:

            confusion_details.append({
                "Loai_loi_thuc_te": actual_class,
                "Loai_loi_du_doan": predicted_class,
                "So_luong_nham": int(cm[i, j])
            })

confusion_df = pd.DataFrame(confusion_details)

if not confusion_df.empty:

    confusion_df = confusion_df.sort_values(
        by="So_luong_nham",
        ascending=False
    )

confusion_df.to_csv(
    "results/phan_tich_loi_nham.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nCÁC CẶP LỚP LỖI DỄ NHẦM:")
print(confusion_df.head(10).to_string(index=False))


# ============================================================
# 15. THÔNG BÁO HOÀN THÀNH
# ============================================================

print("\n" + "=" * 60)
print("HOÀN THÀNH HUẤN LUYỆN!")
print("=" * 60)

print("Mô hình tốt nhất:", best_name)

print("\nCác file mô hình đã lưu trong thư mục models:")

for filename in os.listdir("models"):
    print("-", filename)

print("\nCác kết quả đã lưu trong thư mục results.")

print("\nBây giờ có thể chạy giao diện bằng lệnh:")
print("python -m streamlit run app.py")
