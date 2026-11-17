import pandas as pd
from ucimlrepo import fetch_ucirepo

# 1. Tải trực tiếp dữ liệu thô từ kho UCI
print("=" * 60)
print("ĐANG TẢI DỮ LIỆU TỪ UCI REPOSITORY...")
dataset = fetch_ucirepo(id=198)

X = dataset.data.features.copy()  # 27 đặc trưng
y_onehot = dataset.data.targets.copy()  # 7 cột One-Hot nhãn

print(f"Kích thước dữ liệu gốc: {X.shape[0]} mẫu, {X.shape[1]} đặc trưng")
print(f"Số cột nhãn One-Hot: {y_onehot.shape[1]} cột")

# 2. Gom 7 cột One-Hot thành 1 cột nhãn duy nhất
y_onehot.columns = [str(c).strip() for c in y_onehot.columns]
y = y_onehot.idxmax(axis=1).astype(str)
y.name = "Loai_loi"

# 3. Gộp thành 1 DataFrame hoàn chỉnh
df = X.copy()
df["Loai_loi"] = y

# ==========================================================
# CÁC LỆNH XEM CHI TIẾT BỘ DỮ LIỆU
# ==========================================================

print("\n" + "=" * 60)
print("1. 5 DÒNG ĐẦU TIÊN CỦA BỘ DỮ LIỆU (HEAD):")
print("=" * 60)
pd.set_option("display.max_columns", None)  # Hiển thị tất cả các cột
print(df.head())

print("\n" + "=" * 60)
print("2. THÔNG TIN CÁC CỘT VÀ KIỂU DỮ LIỆU (INFO):")
print("=" * 60)
print(df.info())

print("\n" + "=" * 60)
print("3. PHÂN BỐ CÁC LOẠI LỖI (TARGET DISTRIBUTION):")
print("=" * 60)
print(df["Loai_loi"].value_counts())

print("\n" + "=" * 60)
print("4. BẢNG THỐNG KÊ MÔ TẢ CÁC ĐẶC TRƯNG (DESCRIBE):")
print("=" * 60)
print(df.describe().T[["mean", "std", "min", "50%", "max"]])