
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

steel_data = fetch_ucirepo(id=198)

X = steel_data.data.features.copy()
y_raw = steel_data.data.targets.copy()

print("Kích thước dữ liệu:", X.shape)
print("Kích thước nhãn:", y_raw.shape)

# Kiểm tra dữ liệu thiếu
print("\nSố lượng giá trị thiếu:", X.isnull().sum().sum())

# Kiểm tra dữ liệu vô hạn
X = X.replace([np.inf, -np.inf], np.nan)

# Loại bỏ những dòng có dữ liệu thiếu ở đặc trưng
valid_rows = X.notna().all(axis=1)
X = X.loc[valid_rows].copy()
y_raw = y_raw.loc[valid_rows].copy()

# Chuyển nhãn One-Hot thành nhãn phân loại
if y_raw.shape[1] > 1:
    y_raw = y_raw.astype(int)

    if not y_raw.isin([0, 1]).all().all():
        raise ValueError("Các cột nhãn phải chứa giá trị 0 hoặc 1.")

    if not (y_raw.sum(axis=1) == 1).all():
        raise ValueError(
            "Mỗi dòng phải có đúng một nhãn lỗi bằng 1."
        )

    y = y_raw.idxmax(axis=1)
else:
    y = y_raw.iloc[:, 0]

y = y.astype(str)

# Chuẩn hóa tên cột đặc trưng
X.columns = X.columns.astype(str)

feature_names = list(X.columns)
class_names = sorted(y.unique())

print("\nSố lượng mẫu:", len(X))
print("Số lượng đặc trưng:", len(feature_names))
print("Số loại lỗi:", len(class_names))

print("\nPHÂN BỐ CÁC LOẠI LỖI:")
print(y.value_counts())


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
# 12. VẼ BIỂU ĐỒ SO SÁNH MÔ HÌNH
# ============================================================

metrics_to_plot = [
    "Test_Accuracy",
    "Test_Precision_Macro",
    "Test_Recall_Macro",
    "Test_F1_Macro",
    "Test_AUC_Macro_OVR"
]

for metric in metrics_to_plot:

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=results_df,
        x="Mo_hinh",
        y=metric
    )

    plt.title("So sánh mô hình - " + metric)
    plt.xlabel("Mô hình")
    plt.ylabel(metric)
    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(
        f"results/so_sanh_{metric.lower()}.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 13. VẼ ROC CURVE CHO MÔ HÌNH TỐT NHẤT
# ============================================================

best_prob = test_probabilities[best_name]

best_model_classes = trained_models[best_name].classes_

y_test_binary = label_binarize(
    y_test,
    classes=best_model_classes
)

plt.figure(figsize=(10, 8))

for i, class_name in enumerate(best_model_classes):

    fpr, tpr, _ = roc_curve(
        y_test_binary[:, i],
        best_prob[:, i]
    )

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f"{class_name} (AUC = {roc_auc:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random"
)

plt.title("ROC Curve - " + best_name)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right", fontsize=8)

plt.tight_layout()

plt.savefig(
    "results/roc_curve_best_model.png",
    dpi=300
)

plt.close()


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
