import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ===============================
# PATH
# ===============================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = (
    BASE_DIR
    / "results"
    / "forecast_cuaca_4lokasi_dasarian_6bulan.csv"
)


# ===============================
# CONFIG
# ===============================

st.set_page_config(
    page_title="AI Weather Forecast",
    page_icon="🌦️",
    layout="wide"
)


# ===============================
# HEADER
# ===============================

st.markdown(
    """
    <h1>🌦️ AI Weather Forecast PT Garam - 4 Lokasi</h1>
    <p>
    Prediksi Cuaca Jangka Menengah Menggunakan Transformer Deep Learning
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()


# ===============================
# LOAD DATA
# ===============================

@st.cache_data
def load_data():

    df = pd.read_csv(DATA_FILE)

    return df


df = load_data()



# ===============================
# BUAT PERIODE OTOMATIS
# ===============================

bulan_indo = {
    1:"Januari",
    2:"Februari",
    3:"Maret",
    4:"April",
    5:"Mei",
    6:"Juni",
    7:"Juli",
    8:"Agustus",
    9:"September",
    10:"Oktober",
    11:"November",
    12:"Desember"
}


# mulai setelah bulan sekarang
sekarang = pd.Timestamp.now()

tahun = sekarang.year
bulan = sekarang.month + 1


if bulan > 12:
    bulan = 1
    tahun += 1


periode = []

bulan_temp = bulan
tahun_temp = tahun


for i in range(len(df)):

    dasarian = (i % 3) + 1


    periode.append(
        f"{bulan_indo[bulan_temp]} {tahun_temp} D{dasarian}"
    )


    if dasarian == 3:

        bulan_temp += 1

        if bulan_temp > 12:
            bulan_temp = 1
            tahun_temp += 1


df["Periode"] = periode



# ===============================
# KONVERSI SUHU
# ===============================

df["Suhu (°C)"] = df["t2m"] - 273.15

df["Titik Embun (°C)"] = df["d2m"] - 273.15



# ===============================
# PILIH LOKASI
# ===============================

lokasi_list = sorted(
    df["lokasi"].unique()
)


lokasi_pilihan = st.multiselect(
    "📍 Pilih lokasi prediksi",
    options=lokasi_list,
    default=lokasi_list
)


if not lokasi_pilihan:

    st.warning(
        "Pilih minimal satu lokasi"
    )

    st.stop()


df_filtered = df[
    df["lokasi"].isin(lokasi_pilihan)
]



# ===============================
# INFO CARD
# ===============================

c1,c2,c3,c4 = st.columns(4)


with c1:
    st.metric(
        "📍 Lokasi",
        len(lokasi_pilihan)
    )


with c2:
    st.metric(
        "🤖 Model",
        "Transformer"
    )


with c3:
    st.metric(
        "📅 Forecast",
        "6 Bulan"
    )


with c4:
    st.metric(
        "🌧️ Resolusi",
        "18 Dasarian"
    )



st.divider()



# ===============================
# TABEL PREDIKSI
# ===============================

st.subheader(
    "📋 Tabel Prediksi Cuaca"
)


st.dataframe(

    df_filtered[
        [
            "lokasi",
            "Periode",
            "tp_dasarian_mm",
            "Suhu (°C)",
            "Titik Embun (°C)",
            "ssrd_rate",
            "si10"
        ]
    ].rename(

        columns={
            "tp_dasarian_mm":
            "Curah Hujan (mm)",

            "ssrd_rate":
            "Radiasi Matahari",

            "si10":
            "Kecepatan Angin (m/s)"
        }

    ),

    use_container_width=True,

    hide_index=True

)



st.divider()



# ===============================
# GRAFIK SEMUA PARAMETER
# ===============================

st.subheader(
    "📈 Grafik Prediksi Semua Parameter Cuaca"
)



variables = [

    (
        "tp_dasarian_mm",
        "Curah Hujan (mm)"
    ),

    (
        "Suhu (°C)",
        "Suhu Udara (°C)"
    ),

    (
        "Titik Embun (°C)",
        "Titik Embun (°C)"
    ),

    (
        "ssrd_rate",
        "Radiasi Matahari"
    ),

    (
        "si10",
        "Kecepatan Angin (m/s)"
    )

]



# layout 2 kolom

for i in range(0, len(variables), 2):

    cols = st.columns(2)


    for j in range(2):

        index = i+j


        if index < len(variables):

            var, judul = variables[index]


            with cols[j]:


                fig, ax = plt.subplots(
                    figsize=(6,3)
                )


                for lokasi in lokasi_pilihan:

                    df_lok = (
                        df_filtered[
                            df_filtered["lokasi"] == lokasi
                        ]
                        .reset_index(drop=True)
                    )


                    ax.plot(

                        range(len(df_lok)),

                        df_lok[var],

                        marker="o",

                        label=lokasi

                    )


                ax.set_xticks(
                    range(len(df_lok))
                )


                ax.set_xticklabels(

                    df_lok["Periode"],

                    rotation=45,

                    ha="right",

                    fontsize=8

                )


                ax.set_title(
                    f"{judul} - 6 Bulan"
                )


                ax.set_xlabel(
                    "Periode"
                )


                ax.set_ylabel(
                    judul
                )


                ax.grid(
                    alpha=0.3
                )


                ax.legend(
                    fontsize=7
                )


                st.pyplot(fig)



# ===============================
# CATATAN
# ===============================

st.caption(
    "Model Transformer menggunakan data historis SEAS5 2021-2025 "
    "dengan resolusi dasarian. Prediksi ditampilkan untuk horizon "
    "6 bulan (18 dasarian)."
)