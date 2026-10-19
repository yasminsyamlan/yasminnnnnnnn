from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib


# ==================================================
# PATH
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[1]


# CATATAN PERUBAHAN DARI VERSI 1 LOKASI:
# Versi lama membaca "seas5_karanganyar_2021_2025.csv" (hasil 01_prepare_seas5.py,
# 1 lokasi saja). Sekarang kita langsung memakai CSV gabungan 4 lokasi yang sudah
# dibersihkan & di-agregasi di luar pipeline ini (tidak perlu proses .nc lagi).
INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "SEAS5_GARAM_2021_2025_AGREGAT.csv"
)


OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)


MODEL_DIR = (
    BASE_DIR
    / "models"
)


MODEL_DIR.mkdir(
    exist_ok=True
)


# ==================================================
# LOAD DATA
# ==================================================

print("Membaca data SEAS5 4 lokasi...")


df = pd.read_csv(
    INPUT_FILE
)


# ==================================================
# VALIDASI DATA (WAJIB SEBELUM WINDOWING)
# ==================================================

kolom_wajib = ["tahun", "bulan", "dasarian", "lokasi",
               "tp_dasarian_mm", "t2m", "d2m", "ssrd_rate", "si10"]

hilang = [k for k in kolom_wajib if k not in df.columns]
if hilang:
    raise ValueError(f"Kolom berikut tidak ditemukan di file input: {hilang}")

if df[kolom_wajib].isna().any().any():
    raise ValueError("Masih ada missing value di data. Bersihkan dulu sebelum preprocessing.")

dup = df.duplicated(subset=["tahun", "bulan", "dasarian", "lokasi"]).sum()
if dup > 0:
    raise ValueError(f"Ditemukan {dup} baris duplikat (tahun, bulan, dasarian, lokasi).")


# Urutkan per lokasi lalu waktu (PENTING: windowing di bawah mengasumsikan
# data sudah kronologis per lokasi)

df = df.sort_values(
    ["lokasi", "tahun", "bulan", "dasarian"]
).reset_index(drop=True)


print(df.head())

print(
    "Jumlah data dasarian (semua lokasi):",
    len(df)
)

print(
    "Jumlah lokasi:",
    df["lokasi"].nunique(),
    "->",
    sorted(df["lokasi"].unique())
)

print(df["lokasi"].value_counts())


# ==================================================
# 5 VARIABEL CUACA (FITUR YANG DIPREDIKSI)
# ==================================================

features = [
    "tp_dasarian_mm",
    "t2m",
    "d2m",
    "ssrd_rate",
    "si10"
]


# ==================================================
# FITUR WAKTU SIKLIKAL (BULAN & DASARIAN)
# ==================================================
# Kenapa perlu: bulan 12 secara musiman "dekat" dengan bulan 1 (pergantian
# tahun), tapi kalau ditulis sebagai angka biasa (12 vs 1) modelnya menganggap
# itu jauh. sin/cos membuat siklus tahunan & bulanan terbaca dengan benar
# oleh transformer.

df["bulan_sin"] = np.sin(2 * np.pi * df["bulan"] / 12)
df["bulan_cos"] = np.cos(2 * np.pi * df["bulan"] / 12)

df["dasarian_sin"] = np.sin(2 * np.pi * df["dasarian"] / 3)
df["dasarian_cos"] = np.cos(2 * np.pi * df["dasarian"] / 3)

time_features = ["bulan_sin", "bulan_cos", "dasarian_sin", "dasarian_cos"]


# ==================================================
# ENCODE LOKASI -> ANGKA (UNTUK EMBEDDING DI MODEL)
# ==================================================
# Setiap lokasi diberi id 0..3. Id ini nanti masuk ke Embedding layer di
# 03_train_transformer.py sehingga SATU model bisa belajar pola ke-4 lokasi
# sekaligus, sambil tetap tahu "saya sedang di lokasi mana".

loc_encoder = LabelEncoder()
df["lokasi_id"] = loc_encoder.fit_transform(df["lokasi"])

joblib.dump(
    loc_encoder,
    MODEL_DIR / "location_encoder.pkl"
)

print("\nMapping lokasi -> id:")
for i, nama in enumerate(loc_encoder.classes_):
    print(f"  {i} -> {nama}")


# ==================================================
# NORMALISASI (HANYA 5 VARIABEL CUACA)
# ==================================================
# MinMaxScaler di-fit di seluruh data 4 lokasi SEKALIGUS (bukan per lokasi),
# karena rentang nilai cuaca di 4 titik pegaraman ini berdekatan (masih di
# region Madura yang sama). Satu scaler global, dipakai lagi untuk lokasi
# manapun saat forecast.

print("Normalisasi 5 variabel cuaca...")

scaler = MinMaxScaler()

df[features] = scaler.fit_transform(
    df[features]
)

joblib.dump(
    scaler,
    MODEL_DIR / "scaler.pkl"
)


# ==================================================
# MEMBUAT SEQUENCE TRANSFORMER DASARIAN (PER LOKASI)
# ==================================================

# 1 bulan = 3 dasarian
#
# Input:
# 12 bulan terakhir
# = 36 dasarian
#
# Output:
# 6 bulan kedepan
# = 18 dasarian
#
# PENTING (beda dari versi 1 lokasi): window dibangun TERPISAH per lokasi
# supaya tidak ada "kebocoran" antar lokasi (contoh: 36 dasarian terakhir
# Pamekasan tidak boleh tercampur dengan 18 dasarian pertama Sampang hanya
# karena posisinya bersebelahan di dalam file CSV).

WINDOW = 36
FORECAST = 18

X_seq_list = []      # (36, 5 fitur cuaca + 4 fitur waktu) per sample
X_loc_list = []      # id lokasi (skalar) per sample
y_list = []          # (18, 5 fitur cuaca) per sample

# metadata supaya hasil evaluasi/forecast bisa ditelusuri balik ke
# lokasi & periode aslinya (dipakai 04, 06, 07)
meta_lokasi = []
meta_tahun_awal_target = []
meta_bulan_awal_target = []
meta_dasarian_awal_target = []

seq_cols = features + time_features

for lokasi, df_lok in df.groupby("lokasi", sort=False):

    df_lok = df_lok.sort_values(["tahun", "bulan", "dasarian"]).reset_index(drop=True)

    seq_values = df_lok[seq_cols].to_numpy(dtype="float32")
    feat_values = df_lok[features].to_numpy(dtype="float32")
    lokasi_id = int(df_lok["lokasi_id"].iloc[0])

    tahun_arr = df_lok["tahun"].to_numpy()
    bulan_arr = df_lok["bulan"].to_numpy()
    dasarian_arr = df_lok["dasarian"].to_numpy()

    n = len(df_lok)

    if n < WINDOW + FORECAST:
        print(f"[PERINGATAN] Lokasi '{lokasi}' hanya punya {n} baris, "
              f"butuh minimal {WINDOW + FORECAST}. Lokasi ini dilewati.")
        continue

    for i in range(n - WINDOW - FORECAST + 1):

        X_seq_list.append(seq_values[i:i + WINDOW])
        X_loc_list.append(lokasi_id)
        y_list.append(feat_values[i + WINDOW: i + WINDOW + FORECAST])

        meta_lokasi.append(lokasi)
        meta_tahun_awal_target.append(int(tahun_arr[i + WINDOW]))
        meta_bulan_awal_target.append(int(bulan_arr[i + WINDOW]))
        meta_dasarian_awal_target.append(int(dasarian_arr[i + WINDOW]))


X_seq = np.array(X_seq_list, dtype="float32")   # (N, 36, 9)
X_loc = np.array(X_loc_list, dtype="int32")     # (N,)
y = np.array(y_list, dtype="float32")           # (N, 18, 5)

meta = pd.DataFrame({
    "lokasi": meta_lokasi,
    "tahun_awal_target": meta_tahun_awal_target,
    "bulan_awal_target": meta_bulan_awal_target,
    "dasarian_awal_target": meta_dasarian_awal_target,
})


# ==================================================
# SIMPAN DATA TRAINING
# ==================================================

np.save(OUTPUT_DIR / "X_seq.npy", X_seq)
np.save(OUTPUT_DIR / "X_loc.npy", X_loc)
np.save(OUTPUT_DIR / "y_train.npy", y)

meta.to_csv(OUTPUT_DIR / "window_meta.csv", index=False)

# simpan juga data dasarian (sudah dengan lokasi_id & fitur siklikal) untuk
# dipakai 05_predict_future.py (ambil 36 dasarian TERAKHIR per lokasi)
df.to_csv(OUTPUT_DIR / "dasarian_weather.csv", index=False)


print("==============================")
print("PREPROCESSING DASARIAN 4 LOKASI SELESAI")
print("==============================")

print("X_seq shape:", X_seq.shape)
print("X_loc shape:", X_loc.shape)
print("y shape     :", y.shape)
print("Jumlah window per lokasi:")
print(meta["lokasi"].value_counts())
