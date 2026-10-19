"""Perbarui prediksi dari data terbaru (tanpa training ulang).

Pakai:
  python src/08_update_forecast.py                       # hitung ulang dari data yang ada
  python src/08_update_forecast.py data_baru.csv         # tambah data baru lalu hitung ulang

Format data_baru.csv sama dengan SEAS5_GARAM_2021_2025_AGREGAT.csv:
  tahun,bulan,dasarian,lokasi,tp_dasarian_mm,t2m,d2m,ssrd_rate,si10
"""
import sys

import pandas as pd

import forecast_engine as fe

if len(sys.argv) > 1:
    n = fe.tambah_data_baru(pd.read_csv(sys.argv[1]))
    print(f"Data baru masuk: {n} baris")

f = fe.jalankan_forecast()
print(f"Selesai. {len(f)} baris -> {fe.FORECAST_CSV}")
print(f"Excel -> {fe.EXCEL_FILE}")
print("Periode forecast:", f['Periode'].iloc[0], "s/d", f['Periode'].iloc[-1])
