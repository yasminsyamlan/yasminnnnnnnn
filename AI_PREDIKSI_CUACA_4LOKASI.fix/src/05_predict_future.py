from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model


# =====================================================
# PATH
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = (
    BASE_DIR /
    "data" /
    "processed"
)

MODEL_FILE = (
    BASE_DIR /
    "models" /
    "transformer_weather_model.keras"
)

OUTPUT_DIR = (
    BASE_DIR /
    "results"
)

OUTPUT_DIR.mkdir(exist_ok=True)

FEATURES = ["tp_dasarian_mm", "t2m", "d2m", "ssrd_rate", "si10"]
TIME_FEATURES = ["bulan_sin", "bulan_cos", "dasarian_sin", "dasarian_cos"]
WINDOW = 36
FORECAST = 18

NAMA_BULAN = [
    "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember"
]


# =====================================================
# LOAD MODEL, SCALER, LOCATION ENCODER
# =====================================================

print("Load Transformer model...")
model = load_model(MODEL_FILE)

print("Load scaler & location encoder...")
scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")
loc_encoder = joblib.load(BASE_DIR / "models" / "location_encoder.pkl")


# =====================================================
# LOAD DATA DASARIAN (SUDAH ADA FITUR WAKTU & LOKASI_ID
# DARI 02_preprocessing.py)
# =====================================================

print("Membaca data dasarian...")

df = pd.read_csv(DATA_DIR / "dasarian_weather.csv")


# =====================================================
# FUNGSI BANTUAN: HITUNG PERIODE DASARIAN BERIKUTNYA
# =====================================================

def dasarian_berikutnya(tahun, bulan, dasarian, n_langkah):
    """Hasilkan n_langkah periode (tahun, bulan, dasarian) setelah periode
    yang diberikan. Otomatis pindah bulan setelah Dasarian III, dan pindah
    tahun setelah bulan Desember. Ini menggantikan angka hardcode
    (tahun=2026, bulan=10) di versi 1 lokasi, supaya periode forecast SELALU
    lanjut otomatis dari data terakhir yang ada -- tidak perlu diedit manual
    tiap kali data diupdate.
    """
    hasil = []
    t, b, d = tahun, bulan, dasarian
    for _ in range(n_langkah):
        d += 1
        if d > 3:
            d = 1
            b += 1
            if b > 12:
                b = 1
                t += 1
        hasil.append((t, b, d))
    return hasil


def cyclical_time_features(bulan, dasarian):
    bulan_sin = np.sin(2 * np.pi * bulan / 12)
    bulan_cos = np.cos(2 * np.pi * bulan / 12)
    dasarian_sin = np.sin(2 * np.pi * dasarian / 3)
    dasarian_cos = np.cos(2 * np.pi * dasarian / 3)
    return bulan_sin, bulan_cos, dasarian_sin, dasarian_cos


# =====================================================
# FORECAST PER LOKASI
# =====================================================

semua_forecast = []

for lokasi in loc_encoder.classes_:

    print(f"\n=== Forecast lokasi: {lokasi} ===")

    df_lok = df[df["lokasi"] == lokasi].sort_values(
        ["tahun", "bulan", "dasarian"]
    ).reset_index(drop=True)

    if len(df_lok) < WINDOW:
        print(f"  [PERINGATAN] Data '{lokasi}' kurang dari {WINDOW} baris, dilewati.")
        continue

    # ---------------------------------------------------
    # AMBIL 36 DASARIAN TERAKHIR (ENCODER INPUT)
    # ---------------------------------------------------
    seq_cols = FEATURES + TIME_FEATURES
    last_sequence = df_lok[seq_cols].tail(WINDOW).to_numpy(dtype="float32")

    # catatan: kolom FEATURES di dasarian_weather.csv SUDAH dalam skala
    # 0-1 (hasil scaler.fit_transform di 02_preprocessing.py), jadi di sini
    # TIDAK perlu scaler.transform lagi -- beda dengan versi 1 lokasi yang
    # men-transform ulang data mentah.

    X_seq_future = np.expand_dims(last_sequence, axis=0)          # (1, 36, 9)

    lokasi_id = int(loc_encoder.transform([lokasi])[0])
    X_loc_future = np.array([[lokasi_id]], dtype="int32")          # (1, 1)

    # ---------------------------------------------------
    # FORECAST 18 DASARIAN
    # ---------------------------------------------------
    prediction_scaled = model.predict([X_seq_future, X_loc_future], verbose=0)
    prediction_scaled = prediction_scaled[0]     # (18, 5)

    prediction = scaler.inverse_transform(prediction_scaled)

    # ---------------------------------------------------
    # PERIODE OTOMATIS (LANJUT DARI DATA TERAKHIR)
    # ---------------------------------------------------
    tahun_terakhir = int(df_lok["tahun"].iloc[-1])
    bulan_terakhir = int(df_lok["bulan"].iloc[-1])
    dasarian_terakhir = int(df_lok["dasarian"].iloc[-1])

    periode_list = dasarian_berikutnya(
        tahun_terakhir, bulan_terakhir, dasarian_terakhir, FORECAST
    )

    nama_periode = []
    for (t, b, d) in periode_list:
        romawi = {1: "I", 2: "II", 3: "III"}[d]
        nama_periode.append(f"{NAMA_BULAN[b-1]} {t} Dasarian {romawi}")

    # ---------------------------------------------------
    # SUSUN DATAFRAME HASIL
    # ---------------------------------------------------
    forecast_lok = pd.DataFrame(prediction, columns=FEATURES)
    forecast_lok.insert(0, "lokasi", lokasi)
    forecast_lok.insert(1, "tahun", [p[0] for p in periode_list])
    forecast_lok.insert(2, "bulan", [p[1] for p in periode_list])
    forecast_lok.insert(3, "dasarian", [p[2] for p in periode_list])
    forecast_lok.insert(4, "Periode", nama_periode)

    semua_forecast.append(forecast_lok)

    print(forecast_lok[["Periode"] + FEATURES].to_string(index=False))


# =====================================================
# GABUNG SEMUA LOKASI & SIMPAN
# =====================================================

forecast_gabungan = pd.concat(semua_forecast, ignore_index=True)

output_file = (
    OUTPUT_DIR /
    "forecast_cuaca_4lokasi_dasarian_6bulan.csv"
)

forecast_gabungan.to_csv(output_file, index=False)


print("\n==============================")
print("FORECAST DASARIAN 4 LOKASI SELESAI")
print("==============================")
print("File:", output_file)
print("Total baris:", len(forecast_gabungan), "(4 lokasi x 18 dasarian)")
