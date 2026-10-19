from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ===============================
# PATH
# ===============================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "dasarian_weather.csv"
)

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "transformer_weather_model.keras"
)

SCALER_FILE = (
    BASE_DIR
    / "models"
    / "scaler.pkl"
)

LOC_ENCODER_FILE = (
    BASE_DIR
    / "models"
    / "location_encoder.pkl"
)

FEATURES = ["tp_dasarian_mm", "t2m", "d2m", "ssrd_rate", "si10"]
TIME_FEATURES = ["bulan_sin", "bulan_cos", "dasarian_sin", "dasarian_cos"]
WINDOW = 36

# tahun terakhir dianggap "hold-out" murni (data yang tidak dipakai untuk
# menentukan scaler/encoder, sekadar validasi tambahan berbasis kalender,
# terpisah dari test-split window-based di 04_evaluate_transformer.py)
TAHUN_VALIDASI = 2025


# ===============================
# LOAD
# ===============================

print("Membaca data...")

# CATATAN PERUBAHAN: dasarian_weather.csv sudah memuat fitur waktu siklikal +
# lokasi_id (dihasilkan 02_preprocessing.py) dan kolom FEATURES SUDAH dalam
# skala 0-1 (sudah di-scaler.fit_transform). Jadi di sini TIDAK perlu
# scaler.transform lagi -- beda dari versi 1 lokasi yang membaca data mentah.
df = pd.read_csv(DATA_FILE)

loc_encoder = joblib.load(LOC_ENCODER_FILE)
scaler = joblib.load(SCALER_FILE)

print("Load model...")
model = tf.keras.models.load_model(MODEL_FILE)


# ===============================
# VALIDASI PER LOKASI (BERBASIS TAHUN)
# ===============================
# PENTING (beda dari versi 1 lokasi): versi lama membangun SATU sequence test
# dari potongan tahun 2025 tanpa peduli lokasi (karena memang cuma ada 1
# lokasi). Sekarang WAJIB dipisah per lokasi dulu, baru tiap lokasi dibangun
# sequence-nya sendiri, supaya window 36 dasarian tidak pernah mencampur
# lokasi yang berbeda.

hasil_semua_lokasi = {}

for lokasi in loc_encoder.classes_:

    print(f"\n=== Validasi lokasi: {lokasi} ===")

    df_lok = df[df["lokasi"] == lokasi].sort_values(
        ["tahun", "bulan", "dasarian"]
    ).reset_index(drop=True)

    lokasi_id = int(loc_encoder.transform([lokasi])[0])

    # kita butuh data SEBELUM tahun validasi juga, supaya 36 dasarian
    # pertama di awal tahun validasi tetap punya histori penuh 12 bulan.
    idx_validasi_mulai = df_lok.index[df_lok["tahun"] == TAHUN_VALIDASI]
    if len(idx_validasi_mulai) == 0:
        print(f"  [PERINGATAN] Tidak ada data tahun {TAHUN_VALIDASI} untuk lokasi ini, dilewati.")
        continue

    awal = idx_validasi_mulai[0]
    # kita mulai window dari (awal - WINDOW) supaya prediksi pertama tetap
    # jatuh di tahun validasi
    mulai_window = max(0, awal - WINDOW)

    df_valid = df_lok.iloc[mulai_window:].reset_index(drop=True)

    seq_cols = FEATURES + TIME_FEATURES
    seq_values = df_valid[seq_cols].to_numpy(dtype="float32")
    feat_values = df_valid[FEATURES].to_numpy(dtype="float32")

    n = len(df_valid)

    X_test, y_test = [], []
    for i in range(n - WINDOW):
        X_test.append(seq_values[i:i + WINDOW])
        # validasi one-step-ahead (dasarian pertama setelah window), supaya
        # bisa dibandingkan rapi per titik waktu -- forecast 18-langkah
        # penuh sudah dicek di 04_evaluate_transformer.py & 05
        y_test.append(feat_values[i + WINDOW])

    if len(X_test) == 0:
        print(f"  [PERINGATAN] Data tahun {TAHUN_VALIDASI} tidak cukup untuk window {WINDOW}, dilewati.")
        continue

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_loc_test = np.full((len(X_test), 1), lokasi_id, dtype="int32")

    print("  X test:", X_test.shape)

    pred = model.predict([X_test, X_loc_test], verbose=0)

    # ambil prediksi dasarian pertama saja (langkah ke-0 dari 18 langkah)
    # untuk dibandingkan dengan y_test (one-step-ahead)
    pred = pred[:, 0, :]

    y_test_real = scaler.inverse_transform(y_test)
    pred_real = scaler.inverse_transform(pred)

    hasil_lokasi = {}
    for i, var in enumerate(FEATURES):
        mae = mean_absolute_error(y_test_real[:, i], pred_real[:, i])
        rmse = np.sqrt(mean_squared_error(y_test_real[:, i], pred_real[:, i]))
        r2 = r2_score(y_test_real[:, i], pred_real[:, i])

        print(f"  {var:16s} MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}")

        hasil_lokasi[var] = {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}

    hasil_semua_lokasi[lokasi] = hasil_lokasi


print("\n===================")
print("SELESAI")
print("===================")
