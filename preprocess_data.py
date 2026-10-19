import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

print("--- BẮT ĐẦU TIỀN XỬ LÝ DỮ LIỆU TELCO CHURN (BẢN VIỆT HÓA) ---")

data_dir = r"D:\Tài Liệu\Đồ án TH_KTDL\Data"
input_file_path = os.path.join(data_dir, "Telco-Customer-Churn_VN.csv")

df = pd.read_csv(input_file_path)
print(f"-> Đọc file thành công từ: {input_file_path}")
print(f"-> Dữ liệu ban đầu: {df.shape[0]} dòng, {df.shape[1]} cột.")

df['tong_cuoc_phi'] = pd.to_numeric(df['tong_cuoc_phi'], errors='coerce')
df.dropna(subset=['tong_cuoc_phi'], inplace=True)


numeric_features = ['thoi_gian_gan_bo', 'cuoc_phi_hang_thang', 'tong_cuoc_phi']
categorical_features = [
    'loai_hop_dong', 
    'dich_vu_internet', 
    'phuong_thuc_thanh_toan', 
    'ho_tro_ky_thuat', 
    'bao_mat_truc_tuyen'
]

df_encoded = pd.get_dummies(df[categorical_features], drop_first=True)
feature_df = pd.concat([df[numeric_features], df_encoded], axis=1)
feature_names = feature_df.columns.tolist()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(feature_df)

df_scaled = pd.DataFrame(X_scaled, columns=feature_names)
df_scaled['roi_bo'] = df['roi_bo'].values
df_scaled['ma_khach_hang'] = df['ma_khach_hang'].values

output_clean_path = os.path.join(data_dir, "telco_cleaned.csv")
output_scaled_path = os.path.join(data_dir, "telco_scaled.csv")

df.to_csv(output_clean_path, index=False, encoding='utf-8-sig')
df_scaled.to_csv(output_scaled_path, index=False, encoding='utf-8-sig')

joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_names, "feature_names.pkl")

print("--- HOÀN THÀNH TIỀN XỬ LÝ ---")
print(f"-> Đã xuất file đối chứng: {output_clean_path}")
print(f"-> Đã xuất file ma trận chuẩn hóa: {output_scaled_path}")
print("-> Đã lưu 'scaler.pkl' và 'feature_names.pkl' thành công!")