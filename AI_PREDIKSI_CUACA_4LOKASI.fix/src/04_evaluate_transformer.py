from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import matplotlib.pyplot as plt

import json


# =====================================================
# PATH
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "transformer_weather_model.keras"
)

RESULT_DIR = (
    BASE_DIR
    / "results"
    / "evaluasi"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FEATURES = ["tp_dasarian_mm", "t2m", "d2m", "ssrd_rate", "si10"]


# =====================================================
# LOAD MODEL
# =====================================================

print("Load Transformer model...")

model = tf.keras.models.load_model(
    MODEL_FILE
)


# =====================================================
# LOAD DATA DASARIAN (4 LOKASI)
# =====================================================

print("Load data dasarian...")

X_seq = np.load(DATA_DIR / "X_seq.npy")
X_loc = np.load(DATA_DIR / "X_loc.npy")
y_true = np.load(DATA_DIR / "y_train.npy")

meta = pd.read_csv(DATA_DIR / "window_meta.csv")


print("======================")
print("DATA")
print("======================")

print("X_seq shape:", X_seq.shape)
print("X_loc shape:", X_loc.shape)
print("y shape    :", y_true.shape)


# =====================================================
# GUNAKAN TEST SPLIT YANG SAMA DENGAN TRAINING
# =====================================================
# PENTING (perbaikan dari versi 1 lokasi): skrip lama mengevaluasi model di
# SELURUH data (termasuk data yang sudah dipakai untuk training -> hasil MAE
# jadi terlalu optimis / tidak representatif). Di sini kita evaluasi HANYA
# pada 20% data terakhir tiap lokasi (window paling akhir waktu), yaitu
# bagian yang TIDAK dipakai untuk update bobot model saat training.

test_idx = []
for lokasi in meta["lokasi"].unique():
    idx_lokasi = meta.index[meta["lokasi"] == lokasi].to_numpy()
    split = int(len(idx_lokasi) * 0.8)
    test_idx.extend(idx_lokasi[split:])

test_idx = np.array(sorted(test_idx))

X_seq_test = X_seq[test_idx]
X_loc_test = X_loc[test_idx]
y_test = y_true[test_idx]
meta_test = meta.loc[test_idx].reset_index(drop=True)

print("\nJumlah data test (20% akhir tiap lokasi):", len(test_idx))
print(meta_test["lokasi"].value_counts())


# =====================================================
# PREDIKSI
# =====================================================

print("======================")
print("Melakukan prediksi...")
print("======================")

y_pred = model.predict(
    [X_seq_test, X_loc_test]
)

print(
    "Prediksi shape:",
    y_pred.shape
)


# =====================================================
# METRIK EVALUASI - GABUNGAN (SEMUA LOKASI)
# =====================================================

y_true_flat = y_test.reshape(-1)
y_pred_flat = y_pred.reshape(-1)

mae = mean_absolute_error(y_true_flat, y_pred_flat)
rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
r2 = r2_score(y_true_flat, y_pred_flat)

print("======================")
print("HASIL EVALUASI DASARIAN (GABUNGAN 4 LOKASI, DATA TEST)")
print("======================")
print("MAE  :", mae)
print("RMSE :", rmse)
print("R2   :", r2)


# =====================================================
# METRIK EVALUASI - PER LOKASI & PER VARIABEL
# =====================================================
# PENTING (baru, tidak ada di versi 1 lokasi): karena sekarang 4 lokasi
# digabung dalam 1 model, WAJIB dicek performa TIAP lokasi secara terpisah.
# Ada kemungkinan model lebih bagus di satu lokasi dan kurang bagus di
# lokasi lain (misal karena pola cuaca lebih variatif).

hasil_per_lokasi = {}

print("\n======================")
print("HASIL PER LOKASI & PER VARIABEL")
print("======================")

for lokasi in sorted(meta_test["lokasi"].unique()):
    mask = (meta_test["lokasi"] == lokasi).to_numpy()
    y_true_lok = y_test[mask]     # (n, 18, 5)
    y_pred_lok = y_pred[mask]

    print(f"\n--- {lokasi} ---")
    hasil_per_lokasi[lokasi] = {}

    for i, var in enumerate(FEATURES):
        yt = y_true_lok[:, :, i].reshape(-1)
        yp = y_pred_lok[:, :, i].reshape(-1)

        mae_v = mean_absolute_error(yt, yp)
        rmse_v = np.sqrt(mean_squared_error(yt, yp))
        r2_v = r2_score(yt, yp)

        print(f"  {var:16s} MAE={mae_v:.4f}  RMSE={rmse_v:.4f}  R2={r2_v:.4f}")

        hasil_per_lokasi[lokasi][var] = {
            "MAE": float(mae_v),
            "RMSE": float(rmse_v),
            "R2": float(r2_v),
        }


# =====================================================
# SIMPAN HASIL
# =====================================================

hasil = {
    "model": "Transformer (4 lokasi, location embedding)",
    "resolusi": "Dasarian",
    "forecast_step": "18 dasarian",
    "evaluasi_pada": "20% data terakhir tiap lokasi (test, tidak dipakai training)",
    "gabungan_4_lokasi": {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    },
    "per_lokasi": hasil_per_lokasi,
}

with open(RESULT_DIR / "evaluasi_transformer_dasarian.json", "w") as f:
    json.dump(hasil, f, indent=4)


# =====================================================
# GRAFIK AKTUAL VS PREDIKSI (PER LOKASI)
# =====================================================

for lokasi in sorted(meta_test["lokasi"].unique()):
    mask = (meta_test["lokasi"] == lokasi).to_numpy()
    yt = y_test[mask].reshape(-1)
    yp = y_pred[mask].reshape(-1)

    plt.figure(figsize=(12, 5))
    plt.plot(yt[:200], label="Aktual")
    plt.plot(yp[:200], label="Prediksi")
    plt.title(f"Evaluasi Transformer Dasarian - {lokasi}\nAktual vs Prediksi (data test)")
    plt.xlabel("Urutan Dasarian (dinormalisasi, gabungan semua variabel)")
    plt.ylabel("Nilai Normalisasi (0-1)")
    plt.legend()
    plt.grid()
    plt.savefig(
        RESULT_DIR / f"aktual_vs_prediksi_dasarian_{lokasi}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


print("\n======================")
print("Evaluasi selesai")
print("======================")
print("Hasil JSON  :", RESULT_DIR / "evaluasi_transformer_dasarian.json")
print("Grafik (per lokasi) tersimpan di:", RESULT_DIR)
