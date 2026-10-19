from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# =====================================================
# PATH
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "results"
    / "forecast_cuaca_4lokasi_dasarian_6bulan.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "grafik_dasarian"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================
# LOAD DATA
# =====================================================

print("Membaca forecast dasarian 4 lokasi...")

df = pd.read_csv(INPUT_FILE)

print(df.head())


# =====================================================
# INDEX DASARIAN (PER LOKASI, SUPAYA SUMBU-X SEJAJAR)
# =====================================================

df["Index"] = df.groupby("lokasi").cumcount() + 1


# =====================================================
# LABEL PERIODE (DARI KOLOM Periode, SEKALI SAJA -
# SAMA UNTUK SEMUA LOKASI KARENA HORIZON FORECAST SAMA)
# =====================================================

label_x = []
periode_referensi = df[df["lokasi"] == df["lokasi"].iloc[0]]["Periode"].tolist()

for p in periode_referensi:
    if "Dasarian I" in p and "Dasarian II" not in p and "Dasarian III" not in p:
        d = "D-I"
    elif "Dasarian II" in p:
        d = "D-II"
    else:
        d = "D-III"

    bulan = p.split()[0][:3]
    tahun = p.split()[1]

    label_x.append(f"{bulan}\n{tahun}\n{d}")


# =====================================================
# VARIABEL CUACA
# =====================================================

variables = {
    "tp_dasarian_mm": ("Curah Hujan", "mm"),
    "t2m": ("Suhu Udara", "Kelvin"),
    "d2m": ("Kelembapan Udara", "Kelvin"),
    "ssrd_rate": ("Radiasi Matahari", "W/m\u00b2"),
    "si10": ("Kecepatan Angin", "m/s"),
}

lokasi_list = sorted(df["lokasi"].unique())
warna = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


# =====================================================
# GRAFIK OVERLAY: BANDINGKAN 4 LOKASI DALAM 1 VARIABEL
# =====================================================
# BARU (tidak ada di versi 1 lokasi): karena sekarang ada 4 titik, grafik
# yang paling berguna untuk analisis adalah membandingkan pola ke-4 lokasi
# sekaligus per variabel cuaca, bukan cuma satu lokasi tunggal.

for col, info in variables.items():

    title = info[0]
    satuan = info[1]

    plt.figure(figsize=(14, 6))

    for i, lokasi in enumerate(lokasi_list):
        df_lok = df[df["lokasi"] == lokasi].sort_values("Index")
        plt.plot(
            df_lok["Index"],
            df_lok[col],
            marker="o",
            linewidth=2,
            label=lokasi,
            color=warna[i % len(warna)]
        )

    plt.title(
        f"Prediksi {title}\n"
        "Perbandingan 4 Lokasi - 6 Bulan (18 Dasarian)",
        fontsize=14
    )

    plt.xlabel("Periode Dasarian")
    plt.ylabel(f"{title} ({satuan})")

    plt.xticks(
        range(1, len(label_x) + 1),
        label_x,
        rotation=45,
        ha="right",
        fontsize=8
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{col}_forecast_dasarian_perbandingan.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =====================================================
# GRAFIK PER LOKASI (SAMA SEPERTI VERSI 1 LOKASI, TAPI
# SEKARANG DIULANG UNTUK KE-4 LOKASI, DISIMPAN DI
# SUBFOLDER MASING-MASING)
# =====================================================

for lokasi in lokasi_list:

    lokasi_dir = OUTPUT_DIR / lokasi
    lokasi_dir.mkdir(exist_ok=True)

    df_lok = df[df["lokasi"] == lokasi].sort_values("Index")

    for col, info in variables.items():

        title = info[0]
        satuan = info[1]

        plt.figure(figsize=(14, 6))

        plt.plot(
            df_lok["Index"],
            df_lok[col],
            marker="o",
            linewidth=2
        )

        plt.title(
            f"Prediksi {title}\n"
            f"{lokasi} - 6 Bulan (18 Dasarian)",
            fontsize=14
        )

        plt.xlabel("Periode Dasarian")
        plt.ylabel(f"{title} ({satuan})")

        plt.xticks(
            range(1, len(label_x) + 1),
            label_x,
            rotation=45,
            ha="right",
            fontsize=8
        )

        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            lokasi_dir / f"{col}_forecast_dasarian.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

print("==============================")
print("VISUALISASI SELESAI")
print("==============================")
print("Grafik perbandingan 4 lokasi:", OUTPUT_DIR)
print("Grafik per lokasi (subfolder):", [str(OUTPUT_DIR / l) for l in lokasi_list])
