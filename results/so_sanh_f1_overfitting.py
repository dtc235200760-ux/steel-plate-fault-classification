import io
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Khởi tạo dữ liệu từ bảng kết quả
csv_data = """Mo_hinh,CV_F1_Macro,Tham_so_tot_nhat,Train_Accuracy,Train_Precision_Macro,Train_Recall_Macro,Train_F1_Macro,Train_AUC_Macro_OVR,Test_Accuracy,Test_Precision_Macro,Test_Recall_Macro,Test_F1_Macro,Test_AUC_Macro_OVR
Random Forest,0.7947133009412782,"{'max_depth': 20, 'min_samples_leaf': 1, 'n_estimators': 250}",0.979381443298969,0.975788381162077,0.9915074309978769,0.9832630436274729,0.9999595154078084,0.8082191780821918,0.821068775184998,0.8330099728477209,0.8250123971221388,0.9674490805816282
Gradient Boosting,0.7792386743713392,"{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 150}",0.9955817378497791,0.9977678402284553,0.9973598342934427,0.9975613974403321,0.9999763460323862,0.8013698630136986,0.8311209670627723,0.7930685401774088,0.8102041949400854,0.9646626163991673
SVM,0.7628427686056023,"{'model__C': 5, 'model__gamma': 'scale', 'model__kernel': 'rbf'}",0.8269513991163475,0.8212061996482278,0.9056342779723071,0.852441204135952,0.9822172816099679,0.7534246575342466,0.7673470689462161,0.8251862392143071,0.7877268220086334,0.9568383115755987
KNN,0.7486988972353898,"{'model__n_neighbors': 7, 'model__p': 1, 'model__weights': 'distance'}",1.0,1.0,1.0,1.0,1.0,0.7773972602739726,0.7660965955922243,0.7970702714916721,0.7741699482702291,0.9530689997696914
Extra Trees,0.7954634437818546,"{'max_depth': 20, 'min_samples_leaf': 2, 'n_estimators': 250}",0.9845360824742269,0.9646491537639158,0.9932074204774584,0.9782750567475578,0.999938155751664,0.7705479452054794,0.7478710102446167,0.8085080952351811,0.7724882498022244,0.9669071406508454
"""

# Tạo biến DataFrame 'df'
df = pd.read_csv(io.StringIO(csv_data))

# 2. Đổi định dạng dữ liệu F1-Macro
f1_cols = ["CV_F1_Macro", "Train_F1_Macro", "Test_F1_Macro"]
df_f1 = df.melt(
    id_vars=["Mo_hinh"],
    value_vars=f1_cols,
    var_name="Tập dữ liệu",
    value_name="F1_Macro",
)

df_f1["Tập dữ liệu"] = df_f1["Tập dữ liệu"].map(
    {
        "CV_F1_Macro": "Cross-Validation F1",
        "Train_F1_Macro": "Train F1",
        "Test_F1_Macro": "Test F1",
    }
)

# 3. Vẽ biểu đồ
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 5))
ax = sns.barplot(
    data=df_f1, x="Mo_hinh", y="F1_Macro", hue="Tập dữ liệu", palette="Set2"
)

plt.title(
    "SO SÁNH ĐIỂM F1-MACRO (CV vs TRAIN vs TEST)",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
plt.xlabel("Mô hình", fontsize=11, fontweight="bold")
plt.ylabel("Điểm F1-Macro", fontsize=11, fontweight="bold")
plt.ylim(0.7, 1.05)
plt.legend(title="")

# Hiển thị điểm số trên đỉnh cột
for p in ax.patches:
    val = p.get_height()
    if val > 0:
        ax.annotate(
            f"{val:.3f}",
            (p.get_x() + p.get_width() / 2.0, val),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 3),
            textcoords="offset points",
        )

plt.tight_layout()
# Lưu biểu đồ vào thư mục results
plt.savefig("results/so_sanh_f1_overfitting.png", dpi=300)
print(
    "Đã vẽ và lưu biểu đồ thành công vào results/so_sanh_f1_overfitting.png!"
)
plt.show()