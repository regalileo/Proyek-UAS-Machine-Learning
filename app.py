import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. MEMUAT MODEL DAN OBJEK-OBJEK PENTING DARI FILE ---
# Fungsi ini dijalankan sekali saat aplikasi pertama kali dimuat.
# Menggunakan cache agar tidak memuat ulang file setiap kali ada interaksi.
@st.cache_resource
def load_predict():
    """Memuat predict yang tersimpan (model, encoder, dll.)"""
    try:
        predict = joblib.load("umkm_predict.joblib")
        return predict
    except FileNotFoundError:
        return None

predict = load_predict()

if predict is None:
    st.error("File model 'umkm_predict.joblib' tidak ditemukan. Pastikan Anda sudah menjalankan notebook untuk menyimpan modelnya terlebih dahulu.")
    st.stop()

# Unpack artifacts ke dalam variabel masing-masing
model = predict['model']
le = predict['label_encoder']
train_cols = predict['train_columns']
df_cleaned_for_opts = predict['df_cleaned']
min_values = predict['min_values']


# --- 2. FUNGSI PREPROCESSING DAN PREDIKSI ---
def process_and_predict(df_input, model, le, train_cols, min_values):
    """
    Fungsi ini mengambil DataFrame input, memprosesnya, dan mengembalikan hasil prediksi.
    """
    # Salin input agar tidak mengubah data asli
    df_to_predict = df_input.copy()

    # Hitung aturan bisnis untuk fitur rekayasa
    if df_to_predict['omset'].iloc[0] > 0:
        aturan_profit_margin = df_to_predict['laba'].iloc[0] / df_to_predict['omset'].iloc[0]
    else:
        aturan_profit_margin = 0

    if df_to_predict['aset'].iloc[0] > 0:
        aturan_asset_turnover = df_to_predict['omset'].iloc[0] / df_to_predict['aset'].iloc[0]
    else:
        aturan_asset_turnover = 0
    
    # Tambahkan fitur rekayasa
    df_to_predict['profit_margin'] = aturan_profit_margin
    df_to_predict['asset_turnover'] = aturan_asset_turnover

    # Transformasi log
    log_transform_cols = ['aset', 'omset', 'biaya_karyawan', 'kapasitas_produksi']
    for col in log_transform_cols:
        if col in min_values:
            min_val = min_values[col]
            df_to_predict[col] = df_to_predict[col] - min_val + 1
        df_to_predict[col] = np.log1p(df_to_predict[col])

    df_to_predict['lama_usaha'] = 2025 - df_to_predict['tahun_berdiri']
    df_to_predict['total_tenaga_kerja'] = df_to_predict['tenaga_kerja_perempuan'] + df_to_predict['tenaga_kerja_laki_laki']
    
    # Encoding dan penyamaan kolom
    df_to_predict = pd.get_dummies(df_to_predict)
    df_final = df_to_predict.reindex(columns=train_cols, fill_value=0)
    
    # Lakukan Prediksi
    prediction_result = model.predict(df_final)
    prediction_label = le.inverse_transform(prediction_result)[0]

    # Hitung juga status berdasarkan aturan bisnis untuk perbandingan
    skor_aturan = 0
    if df_input['laba'].iloc[0] > 0: skor_aturan += 2
    if df_input['laba'].iloc[0] > df_input['biaya_karyawan'].iloc[0]: skor_aturan += 2
    if aturan_profit_margin > 0.1: skor_aturan += 1
    if aturan_asset_turnover > 1: skor_aturan += 1
    kunci_jawaban = 'Sehat' if skor_aturan >= 4 else 'Tidak Sehat'

    return prediction_label, kunci_jawaban


# --- 3. TAMPILAN APLIKASI WEB STREAMLIT ---
st.title("📊 Aplikasi Prediksi Kesehatan UMKM")
st.markdown(
    """
    Aplikasi ini menggunakan model Machine Learning  
    Dengan Tujuan untuk memprediksi status kesehatan sebuah usaha berdasarkan data operasional dan keuangan
    """
)

# Panel Input Manual di Sidebar
st.sidebar.header("Masukkan Data Usaha:")

def form_input_pengguna():
    """Menampilkan form di sidebar dan mengembalikan input sebagai DataFrame."""
    jenis_usaha = st.sidebar.selectbox("Jenis Usaha:", options=df_cleaned_for_opts['jenis_usaha'].unique())
    marketplace = st.sidebar.selectbox("Marketplace Utama:", options=df_cleaned_for_opts['marketplace'].unique())
    status_legalitas = st.sidebar.selectbox("Status Legalitas:", options=df_cleaned_for_opts['status_legalitas'].unique())
    
    laba = st.sidebar.number_input("Laba (Rp)", value=5000000, step=100000)
    biaya_karyawan = st.sidebar.number_input("Total Biaya Karyawan (Rp)", value=2000000, step=100000, min_value=0)
    omset = st.sidebar.number_input("Omset Bulanan (Rp)", value=30000000, step=1000000, min_value=0)
    aset = st.sidebar.number_input("Total Aset (Rp)", value=50000000, step=1000000, min_value=0)
    tahun_berdiri = st.sidebar.number_input("Tahun Berdiri", min_value=1980, max_value=2025, value=2020, step=1)
    tk_perempuan = st.sidebar.number_input("Jumlah Tenaga Kerja Perempuan", min_value=0, value=3)
    tk_laki = st.sidebar.number_input("Jumlah Tenaga Kerja Laki-laki", min_value=0, value=3)
    kapasitas_produksi = st.sidebar.number_input("Kapasitas Produksi per Bulan", min_value=0, value=500)
    jumlah_pelanggan = st.sidebar.number_input("Jumlah Pelanggan Aktif", min_value=0, value=150)

    data = {
        'jenis_usaha': jenis_usaha, 'marketplace': marketplace, 'status_legalitas': status_legalitas,
        'laba': laba, 'biaya_karyawan': biaya_karyawan, 'omset': omset,
        'aset': aset, 'tahun_berdiri': tahun_berdiri, 'tenaga_kerja_perempuan': tk_perempuan,
        'tenaga_kerja_laki_laki': tk_laki, 'kapasitas_produksi': kapasitas_produksi,
        'jumlah_pelanggan': jumlah_pelanggan
    }
    fitur = pd.DataFrame(data, index=[0])
    return fitur

# Panggil fungsi form dan simpan hasilnya
df_input_manual = form_input_pengguna()

# Tombol Prediksi
if st.sidebar.button("Prediksi Status Kesehatan"):
    # Panggil fungsi untuk memproses dan mendapatkan hasil
    hasil_prediksi, kunci_jawaban = process_and_predict(df_input_manual, model, le, train_cols, min_values)

    # Tampilkan hasil di area utama
    st.subheader("Hasil Analisis & Prediksi")
    
    st.write(f"Berdasarkan **Aturan Bisnis** yang ditetapkan, status usaha ini adalah: **{kunci_jawaban}**")
    
    if hasil_prediksi == 'Sehat':
        st.success(f"Model Machine Learning memprediksi status usaha Anda: **SEHAT**")
    else:
        st.error(f"Model Machine Learning memprediksi status usaha Anda: **TIDAK SEHAT**")
        
    if hasil_prediksi == kunci_jawaban:
        st.info("💡 **Catatan:** Prediksi model **SESUAI** dengan aturan bisnis.")
    else:
        st.warning("💡 **Catatan:** Prediksi model **BERBEDA** dengan aturan bisnis.")

else:
    # Pesan default di halaman utama
    st.info("Silakan masukkan data di panel sebelah kiri dan klik tombol prediksi untuk melihat hasilnya.")
