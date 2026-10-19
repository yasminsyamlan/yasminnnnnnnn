# Perubahan: AI Prediksi Cuaca 1 Lokasi -> 4 Lokasi

## Ringkasan
Proyek ini adalah update dari `AI PREDIKSI CUACA` (versi 1 lokasi, Karanganyar)
menjadi versi **4 lokasi sekaligus** (Pamekasan_Pandan_Galis,
Sampang_Pangarengan, Sumenep1_Karanganyar, Sumenep_Gersik_Putih), dengan
**satu model Transformer gabungan** (lokasi menjadi fitur input via
embedding), bukan 4 model terpisah.

## Cara pakai (urutan run)
1. `02_preprocessing.py` -- baca `data/processed/SEAS5_GARAM_2021_2025_AGREGAT.csv`,
   hasilkan `X_seq.npy`, `X_loc.npy`, `y_train.npy`, `window_meta.csv`,
   `dasarian_weather.csv`, serta `models/scaler.pkl` dan
   `models/location_encoder.pkl`.
2. `03_train_transformer.py` -- training model, hasil `models/transformer_weather_model.keras`.
3. `04_evaluate_transformer.py` -- evaluasi MAE/RMSE/R2 gabungan & per lokasi
   (di data test yang belum pernah dilihat model).
4. `05_predict_future.py` -- forecast 18 dasarian (6 bulan) ke depan untuk
   ke-4 lokasi, hasil `results/forecast_cuaca_4lokasi_dasarian_6bulan.csv`.
5. `06_visualization.py` -- grafik perbandingan antar lokasi + grafik per
   lokasi, di `results/grafik_dasarian/`.
6. `07_validation_model.py` -- validasi tambahan berbasis tahun (2025)
   per lokasi.
7. `dashboard.py` -- `streamlit run src/dashboard.py`, dengan pemilih lokasi.

**01_prepare_seas5.py TIDAK dipakai** di alur ini karena kamu sudah punya CSV
gabungan 4 lokasi yang bersih (`SEAS5_GARAM_2021_2025_AGREGAT.csv`). File
lama itu masih ada di `src/` untuk referensi kalau suatu saat perlu proses
ulang dari `.nc` mentah.

## Perubahan teknis utama

| Bagian | Versi lama (1 lokasi) | Versi baru (4 lokasi) |
|---|---|---|
| Input data | `seas5_karanganyar_2021_2025.csv` | `SEAS5_GARAM_2021_2025_AGREGAT.csv` (kolom `lokasi`) |
| Encoding lokasi | tidak ada | `LabelEncoder` -> `lokasi_id` -> `Embedding` layer di model |
| Fitur waktu | tidak ada | `bulan_sin/cos`, `dasarian_sin/cos` (musiman) |
| Windowing | 1 seri waktu tunggal | window dibangun **per lokasi**, baru digabung (agar tidak bocor antar lokasi) |
| Split train/val | index tunggal 80/20 | 80/20 **per lokasi**, baru digabung |
| Model | `Input(36,5)` | `Input(36,9)` (5 cuaca + 4 waktu) + `Input(1,)` (id lokasi) -> embedding lokasi digabung ke tiap timestep |
| Evaluasi | di seluruh data (termasuk data training - bias optimis) | di 20% data test per lokasi (belum pernah dilihat model) + breakdown per lokasi |
| Forecast periode | tahun/bulan awal **hardcode** (2026, bulan 10) | otomatis dihitung dari tanggal terakhir di data |
| Visualisasi | 1 lokasi | grafik perbandingan 4 lokasi + grafik per lokasi (subfolder) |
| Dashboard | 1 lokasi tetap | pemilih lokasi (multiselect) |

## Catatan penting soal akurasi
Hasil evaluasi (`results/evaluasi/evaluasi_transformer_dasarian.json`)
menunjukkan **R2 negatif** untuk horizon 6 bulan penuh di data test. Ini
BUKAN bug kode -- ini keterbatasan data:
- Histori cuaca cuma 5 tahun (2021-2025) -> untuk pola musiman/tahunan,
  model idealnya perlu lebih banyak siklus tahun untuk belajar pola yang
  berulang.
- Window test yang tersedia relatif sedikit (~26 window per lokasi).
- Forecast 6 bulan (18 dasarian) ke depan jauh lebih sulit dibanding 1-2
  bulan ke depan; akurasi akan menurun makin jauh horizonnya.

**Saran jika ingin akurasi lebih baik:**
- Tambah histori data (idealnya 10+ tahun) jika tersedia dari sumber SEAS5/ERA5.
- Coba horizon lebih pendek dulu (mis. 1-3 bulan / 3-9 dasarian) untuk validasi
  konsep, baru perpanjang ke 6 bulan.
- Pertimbangkan baseline sederhana (rata-rata klimatologis per bulan/dasarian)
  sebagai pembanding -- kalau Transformer belum jauh lebih baik dari rata-rata
  historis, itu sinyal untuk kembali ke tuning hyperparameter atau menambah data.


---
# Update v2: dashboard time series + Excel + pembaruan otomatis

**Jalankan dashboard:** `pip install -r requirements.txt` lalu `streamlit run src/dashboard.py`

| File | Fungsi |
|---|---|
| `src/forecast_engine.py` | Inti: baca histori, validasi, tambah data baru, forecast 18 dasarian, buat Excel |
| `src/dashboard.py` | Dashboard baru: HANYA grafik curah hujan (historis biru + prediksi oranye putus-putus + garis "Masa Prediksi"). Parameter lain -> tombol download Excel |
| `src/08_update_forecast.py` | Versi command-line dari tombol "perbarui" (bisa dijadwalkan cron/Task Scheduler) |
| `src/dashboard_lama.py` | Dashboard versi sebelumnya (arsip) |
| `results/prediksi_parameter_cuaca.xlsx` | Excel hasil prediksi: sheet "Semua Lokasi" + 1 sheet per lokasi |

**Cara memperbarui (otomatis):** dashboard tidak punya tombol upload lagi. Data cuaca baru masuk ke
`data/processed/SEAS5_GARAM_2021_2025_AGREGAT.csv` (nanti lewat API dari ERP, memanggil
`forecast_engine.tambah_data_baru(df)`; atau manual `python src/08_update_forecast.py data_baru.csv`).
Dashboard mendeteksi sendiri kalau data lebih baru / bulan berganti, lalu menghitung ulang prediksi.

**Tampilan:** grafik hujan di atas, tabel parameter lain (gaya Excel) di bawahnya; halaman memantau data tiap 60 detik.

**Aturan awal prediksi:** selalu mulai BULAN DEPAN. Dasarian yang belum ada data aktualnya sampai akhir
bulan ini diisi sementara dengan rata-rata historis (garis abu-abu putus-putus di grafik) dan otomatis
diganti data aktual begitu masuk. Model tidak dilatih ulang saat update; untuk melatih ulang jalankan 02 -> 03 -> 04.

**Bug dashboard lama yang diperbaiki:** label periode dibuat dari tanggal hari ini (`Timestamp.now()`),
bukan dari periode forecast sebenarnya, sehingga label bisa tidak cocok dengan angkanya.
Sekarang memakai kolom `Periode` dari hasil forecast.
