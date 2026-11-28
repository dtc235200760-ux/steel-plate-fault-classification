
import io
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ============================================================
# 1. KHỞI TẠO DỮ LIỆU TỪ BẢNG KẾT QUẢ
# ============================================================

csv_data = """Mo_hinh,CV_F1_Macro,Tham_so_tot_nhat,Train_Accuracy,Train_Precision_Macro,Train_Recall_Macro,Train_F1_Macro,Train_AUC_Macro_OVR,Val_F1_Macro,Test_Accuracy,Test_Precision_Macro,Test_Recall_Macro,Test_F1_Macro,Test_AUC_Macro_OVR
Random Forest,0.7815819183838354,"{'max_depth': None, 'min_samples_split': 2, 'n_estimators': 200}",1.000000,1.000000,1.000000,1.000000,1.000000,0.803340,0.808219,0.833607,0.799513,0.810690,0.971083
Gradient Boosting,0.7792026845716804,"{'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 200}",0.980118,0.988149,0.988940,0.988541,0.999291,0.803457,0.808219,0.835550,0.783602,0.807204,0.965404
SVM,0.7723342990109826,"{'classifier__C': 5, 'classifier__gamma': 'scale', 'classifier__kernel': 'rbf'}",0.863770,0.887093,0.873242,0.878359,0.985061,0.777102,0.770548,0.814461,0.768439,0.787874,0.958903
KNN,0.7486988972353898,"{'classifier__metric': 'manhattan', 'classifier__n_neighbors': 7, 'classifier__weights': 'distance'}",1.000000,1.000000,1.000000,1.000000,1.000000,0.750403,0.777397,0.766097,0.797070,0.774170,0.953069
Extra Trees,0.7900232106268044,"{'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 100}",1.000000,1.000000,1.000000,1.000000,1.000000,0.805025,0.794521,0.815627,0.793206,0.801600,0.968530
"""

df = pd.read_csv(io.StringIO(csv_data))

# ============================================================
# 2. CHỌN CÁC CỘT F1-MACRO
# ============================================================

f1_cols = [
    "CV_F1_Macro",
    "Train_F1_Macro",
    "Val_F1_Macro",
    "Test_F1_Macro"
]

df_f1 = df.melt(
    id_vars=["Mo_hinh"],
    value_vars=f1_cols,
    var_name="Tập dữ liệu",
    value_name="F1_Macro"
)

# Đổi tên hiển thị trên biểu đồ
df_f1["Tập dữ liệu"] = df_f1["Tập dữ liệu"].map({
    "CV_F1_Macro": "Cross-Validation F1",
    "Train_F1_Macro": "Train F1",
    "Val_F1_Macro": "Validation F1",
    "Test_F1_Macro": "Test F1"
})

# ============================================================
# 3. VẼ BIỂU ĐỒ
# ============================================================

sns.set_theme(style="whitegrid")

plt.figure(figsize=(13, 6))

ax = sns.barplot(
    data=df_f1,
    x="Mo_hinh",
    y="F1_Macro",
    hue="Tập dữ liệu",
    hue_order=[
        "Cross-Validation F1",
        "Train F1",
        "Validation F1",
        "Test F1"
    ],
    palette="Set2"
)

plt.title(
    "SO SÁNH ĐIỂM F1-MACRO "
    "(CROSS-VALIDATION VS TRAIN VS VALIDATION VS TEST)",
    fontsize=13,
    fontweight="bold",
    pad=15
)

plt.xlabel(
    "Mô hình",
    fontsize=11,
    fontweight="bold"
)

plt.ylabel(
    "Điểm F1-Macro",
    fontsize=11,
    fontweight="bold"
)

plt.ylim(0.7, 1.08)

plt.xticks(
    rotation=0,
    ha="center",
    fontsize=10
)

plt.legend(
    title="",
    loc="upper right"
)

# ============================================================
# 4. HIỂN THỊ GIÁ TRỊ TRÊN ĐẦU CỘT
# ============================================================

for p in ax.patches:
    val = p.get_height()

    if pd.notna(val) and val > 0:
        ax.annotate(
            f"{val:.3f}",
            (
                p.get_x() + p.get_width() / 2.0,
                val
            ),
            ha="center",
            va="bottom",
            fontsize=8,
            xytext=(0, 3),
            textcoords="offset points"
        )

plt.tight_layout()

# ============================================================
# 5. LƯU BIỂU ĐỒ
# ============================================================

os.makedirs("results", exist_ok=True)

plt.savefig(
    "results/so_sanh_f1_overfitting.png",
    dpi=300,
    bbox_inches="tight"
)

print(
    "Đã vẽ và lưu biểu đồ thành công tại: "
    "results/so_sanh_f1_overfitting.png"
)

plt.show()
