import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forecast_engine as fe  # noqa: E402

BIRU, ORANYE, ABU, HIJAU = "#3b82f6", "#f97316", "#9ca3af", "#10b981"
BULAN_PENDEK = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
                "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]

st.set_page_config(page_title="Prediksi Curah Hujan PT Garam", page_icon="🌧️", layout="wide")


# ------------------------------------------------------------------ pembaruan otomatis
# Prediksi dihitung ulang sendiri kalau: bulan berganti, atau data histori
# (mis. dari ERP lewat API) lebih baru daripada hasil prediksi terakhir.
if fe.forecast_usang():
    try:
        with st.spinner("Memperbarui prediksi dari data terbaru..."):
            fe.jalankan_forecast()
        st.cache_data.clear()
    except Exception as e:
        st.error(f"Gagal memperbarui prediksi: {e}")
        if not fe.FORECAST_CSV.exists():
            st.stop()


@st.cache_data
def baca_forecast(_key):
    return pd.read_csv(fe.FORECAST_CSV)


@st.cache_data
def baca_histori(_key):
    return fe.load_history()


def tanggal(df):
    """Dasarian -> tanggal tengah dasarian (tgl 5 / 15 / 25)."""
    hari = df["dasarian"].map({1: 5, 2: 15, 3: 25})
    return pd.to_datetime(dict(year=df["tahun"], month=df["bulan"], day=hari))


# ------------------------------------------------------------------ grafik
def buat_grafik(gab, fc, klim=None):
    gab = gab.assign(x=tanggal(gab))
    fc = fc.assign(x=tanggal(fc))
    y = "tp_dasarian_mm"
    aktual = gab[gab["jenis"] == "aktual"]
    isi = gab[gab["jenis"] == "isi"]

    fig = go.Figure()
    if not aktual.empty:
        fig.add_trace(go.Scatter(
            x=aktual["x"], y=aktual[y], mode="lines+markers", name="Data Historis",
            line=dict(color=BIRU, width=2.5), marker=dict(size=6),
            customdata=aktual[["tahun", "bulan", "dasarian"]],
            hovertemplate="Dasarian %{customdata[2]} / %{customdata[1]}-%{customdata[0]}"
                          "<br>%{y:.1f} mm<extra>Aktual</extra>"))

    if not isi.empty:   # belum ada data aktual -> rata-rata historis (sementara)
        xs = ([aktual["x"].iloc[-1]] if not aktual.empty else []) + isi["x"].tolist()
        ys = ([aktual[y].iloc[-1]] if not aktual.empty else []) + isi[y].tolist()
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name="Belum ada data (rata-rata historis)",
            line=dict(color=ABU, width=2, dash="dot"),
            hovertemplate="%{y:.1f} mm<extra>Perkiraan sementara</extra>"))

    ujung = gab.iloc[-1]
    xs = [ujung["x"]] + fc["x"].tolist()
    ys = [ujung[y]] + fc[y].tolist()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines+markers", name="Data Prediksi",
        line=dict(color=ORANYE, width=2.5, dash="dash"),
        marker=dict(size=[0] + [6] * len(fc)),
        text=[""] + fc["Periode"].tolist(),
        hovertemplate="%{text}<br>%{y:.1f} mm<extra>Prediksi</extra>"))

    if klim is not None:
        fig.add_trace(go.Scatter(
            x=fc["x"], y=klim, mode="lines", name="Acuan rata-rata historis",
            line=dict(color=HIJAU, width=1.8, dash="dot"),
            hovertemplate="%{y:.1f} mm<extra>Rata-rata historis</extra>"))

    batas = ujung["x"] + pd.Timedelta(days=5)
    fig.add_shape(type="line", x0=batas, x1=batas, y0=0, y1=1, yref="paper",
                  line=dict(color="#6b7280", width=1, dash="dash"))
    fig.add_annotation(x=batas, y=1.06, yref="paper", text="Masa Prediksi",
                       showarrow=False, font=dict(size=11, color="#4b5563"))

    awal, akhir = gab["x"].min(), fc["x"].max()
    bulan = pd.date_range(awal.replace(day=1), akhir, freq="MS")
    tickvals = [b + pd.Timedelta(days=14) for b in bulan]
    ticktext = [f"{BULAN_PENDEK[b.month-1]}<br>{b.year}" if (b.month == 1 or i == 0)
                else BULAN_PENDEK[b.month - 1] for i, b in enumerate(bulan)]

    fig.update_layout(
        template="plotly_white", height=430, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        yaxis=dict(title="mm", rangemode="tozero", gridcolor="#eef0f3"),
        xaxis=dict(tickmode="array", tickvals=tickvals, ticktext=ticktext, gridcolor="#f3f4f6",
                   range=[awal - pd.Timedelta(days=10), akhir + pd.Timedelta(days=10)]),
        hovermode="closest")
    return fig


def tampil_chart(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:                       # Streamlit versi lama
        st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------------ halaman
st.title("🌧️ Prediksi Curah Hujan 6 Bulan")
st.caption("Transformer time series · resolusi dasarian · 4 lokasi PT Garam")

fc_all = baca_forecast(fe.FORECAST_CSV.stat().st_mtime)
hist_all = baca_histori(fe.RAW_FILE.stat().st_mtime)
_, celah_all = fe.isi_celah(hist_all)

c1, c2 = st.columns([2, 1])
with c1:
    lokasi = st.selectbox("📍 Lokasi", sorted(fc_all["lokasi"].unique()))
with c2:
    riwayat = st.selectbox("Riwayat yang ditampilkan", ["6 bulan", "12 bulan", "24 bulan", "Semua"], index=1)

aktual = hist_all[hist_all["lokasi"] == lokasi].assign(jenis="aktual")
isi = celah_all[celah_all["lokasi"] == lokasi].assign(jenis="isi")
gab = pd.concat([aktual, isi]).sort_values(["tahun", "bulan", "dasarian"]).reset_index(drop=True)
if riwayat != "Semua":
    gab = gab.tail(int(riwayat.split()[0]) * 3).reset_index(drop=True)
fc = fc_all[fc_all["lokasi"] == lokasi].reset_index(drop=True)

if not isi.empty:
    ter = aktual.iloc[-1]
    st.info(
        f"Data aktual terakhir: **{fe.NAMA_BULAN[int(ter['bulan'])-1]} {int(ter['tahun'])}**. "
        f"Dasarian sesudahnya sampai akhir bulan ini belum ada data, jadi diisi sementara dengan "
        f"rata-rata historis. Isian ini otomatis diganti data aktual begitu data baru masuk."
    )

tampil_acuan = st.checkbox("Tampilkan garis acuan rata-rata historis (klimatologi)")
klim = None
if tampil_acuan:
    dasar = aktual.groupby(["bulan", "dasarian"])["tp_dasarian_mm"].mean()
    klim = [dasar[(b, d)] for b, d in zip(fc["bulan"], fc["dasarian"])]

puncak = fc.loc[fc["tp_dasarian_mm"].idxmax()]
ter = aktual.iloc[-1]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Data aktual terakhir", f"{BULAN_PENDEK[int(ter['bulan'])-1]} {int(ter['tahun'])} D{int(ter['dasarian'])}")
m2.metric("Periode prediksi",
          f"{BULAN_PENDEK[int(fc['bulan'].iloc[0])-1]} {int(fc['tahun'].iloc[0])} – "
          f"{BULAN_PENDEK[int(fc['bulan'].iloc[-1])-1]} {int(fc['tahun'].iloc[-1])}")
m3.metric("Total hujan 6 bulan", f"{fc['tp_dasarian_mm'].sum():,.0f} mm")
m4.metric("Dasarian terbasah", f"{puncak['tp_dasarian_mm']:.0f} mm", puncak["Periode"], delta_color="off")

tampil_chart(buat_grafik(gab, fc, klim))

# ------------------------------------------------------------------ tabel parameter lain
st.divider()
st.subheader(f"📊 Parameter cuaca lainnya — {lokasi}")
tabel = fe._tabel_excel(fc).drop(columns=["Tahun", "Bulan", "Dasarian", fe.KOLOM_EXCEL["tp_dasarian_mm"]])
st.dataframe(tabel, hide_index=True, height=38 + 35 * len(tabel))
st.download_button(
    "⬇️ Download Excel (semua lokasi)",
    data=fe.buat_excel(fc_all),
    file_name=f"prediksi_cuaca_{int(fc_all['tahun'].iloc[0])}-{int(fc_all['bulan'].iloc[0]):02d}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# ------------------------------------------------------------------ akurasi
if fe.EVAL_FILE.exists():
    with st.expander("ℹ️ Seberapa akurat model ini?"):
        ev = json.loads(fe.EVAL_FILE.read_text())
        baris = [{"Lokasi": k, "MAE (skala 0-1)": round(v["tp_dasarian_mm"]["MAE"], 3),
                  "R² curah hujan": round(v["tp_dasarian_mm"]["R2"], 2)}
                 for k, v in ev["per_lokasi"].items()]
        st.dataframe(pd.DataFrame(baris), hide_index=True)
        st.caption("R² mendekati/di bawah 0 berarti model belum lebih baik daripada menebak "
                   "rata-rata. Baca prediksi 6 bulan sebagai tren kasar, bukan angka pasti.")


# ------------------------------------------------------------------ pemantauan otomatis
# Tiap 60 detik cek apakah ada data baru / bulan berganti. Kalau ya, halaman
# memuat ulang sendiri dan prediksi dihitung ulang (tanpa perlu refresh manual).
@st.fragment(run_every=60)
def pantau():
    if fe.forecast_usang():
        st.rerun()
    st.caption(f"🔄 Dipantau otomatis · dicek terakhir {pd.Timestamp.now():%d-%m-%Y %H:%M:%S}")


pantau()
