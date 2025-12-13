import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import warnings

# Konfigurasi Halaman (Harus di baris pertama)
st.set_page_config(
    page_title="Sistem Prediksi Kinerja Akademik Siswa",
    layout="wide"
)

warnings.filterwarnings("ignore")

# --- 1. Fungsi Load Model ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load('best_model.pkl')
        return model
    except FileNotFoundError:
        return None

model = load_model()

# --- 2. Sidebar Input (User Interface) ---
st.sidebar.header("📝 Input Data Siswa")
st.sidebar.markdown("Sesuaikan parameter di bawah ini:")
# Grouping Input
with st.sidebar.expander("Faktor Akademik", expanded=True):
    nama = st.text_input("Siapa Nama Kamu ?", value="Siswa Contoh")
    study_hours = st.slider("Berapa Lama Jam Belajar Kamu per hari?", 0.0, 12.0, 3.5, step=0.5)
    attendance = st.slider("Seberapa sering Kehadiran kamu di kelas? (%)", 0.0, 100.0, 90.0)

with st.sidebar.expander("Faktor Gaya Hidup", expanded=True):
    sleep_hours = st.slider(" Berapa jam kamu tidur per malam?", 0.0, 12.0, 7.0, step=0.5)
    mental_health = st.slider(" Bagaimana Kesehatan Mental Kamu? (1-10)", 1, 10, 7, help="1 = Sangat Buruk, 10 = Sangat Baik")
    part_time_job = st.radio("Apakah Bekerja Paruh Waktu?", ("Tidak", "Ya"))

# Encoding Data
ptj_encode = 1 if part_time_job == "Ya" else 0

# --- 3. Main Content ---
st.title("Prediksi Kinerja Akademik Siswa Menggunakan Linear Regression")
st.markdown("Aplikasi berbasis AI untuk memprediksi nilai akhir dan memberikan saran perbaikan kebiasaan belajar.")
st.divider()

# Tombol Prediksi
if st.sidebar.button("Analisis & Prediksi", type="primary"):
    
    # --- Logika Prediksi ---
    input_data = np.array([[study_hours, attendance, mental_health, sleep_hours, ptj_encode]])
    
    if model:
        prediction = model.predict(input_data)[0]
    else:
        # Dummy logic jika file model belum ada (Hanya untuk Demo UI)
        st.warning("⚠️ Model 'best_model.pkl' tidak ditemukan. Menggunakan simulasi logika sederhana.")
        prediction = (study_hours * 2) + (attendance * 0.5) + (mental_health * 1.5) + (sleep_hours * 1) 
        if part_time_job == "Ya": prediction -= 5

    # Clamp nilai antara 0-100
    final_score = max(0, min(100, prediction))

    # --- 4. Tampilan Hasil Dashboard ---
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Hasil Prediksi")
        st.subheader(f"{nama}")
        # Menentukan Status Warna
        if final_score >= 85:
            color = "normal" # Greenish default
            status = "Sangat Baik 🌟"
            pesan = "Luar biasa! Pertahankan kebiasaan baikmu."
        elif final_score >= 70:
            color = "off" 
            status = "Baik 👍"
        else:
            color = "inverse"
            status = "Perlu Perhatian ⚠️"

        st.metric(label="Estimasi Nilai Akhir", value=f"{final_score:.2f}", delta=status)
        
        # Actionable Advice Logic
        st.info("💡 **Saran Peningkatan:**")
        if sleep_hours < 6:
            st.write("- 😴 **Tidur:** Tingkatkan jam tidur minimal 7 jam.")
        if study_hours < 2:
            st.write("- 📚 **Belajar:** Jam belajar mandiri Anda sangat rendah.")
        if attendance < 80:
            st.write("- 🏫 **Kehadiran:** Kehadiran di bawah 80% sangat berisiko.")
        if mental_health < 5:
            st.write("- 🧘 **Kesehatan:** Jangan ragu berkonsultasi dengan konselor.")
        if final_score > 85 and sleep_hours > 6:
            st.write("- ✨ **Pertahankan:** Kebiasaan Anda sudah sangat seimbang!")

    with col2:
        st.subheader("Analisis Gaya Hidup vs Ideal")
        
        # --- 5. Visualisasi Radar Chart ---
        # Normalisasi data agar skalanya sama (0-1) untuk grafik
        categories = ['Jam Belajar', 'Kehadiran', 'Kesehatan Mental', 'Jam Tidur']
        
        # Data User (Dinormalisasi secara kasar untuk visualisasi)
        values_user = [
            min(study_hours/10, 1), 
            attendance/100, 
            mental_health/10, 
            min(sleep_hours/9, 1)
        ]
        
        # Data Ideal (Benchmark)
        values_ideal = [0.6, 0.95, 0.9, 0.88] # Misal: 6 jam belajar, 95% hadir, dst.

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=values_user,
            theta=categories,
            fill='toself',
            name='Kondisi Anda',
            line_color='blue'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=values_ideal,
            theta=categories,
            fill='toself',
            name='Target Ideal',
            line_color='green',
            opacity=0.4
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=400,
            margin=dict(l=40, r=40, t=20, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Silakan atur parameter di sidebar dan tekan tombol **'Analisis & Prediksi'**.")

# --- Tambahkan ini di bawah Radar Chart ---
st.divider()
st.subheader("🔍 Faktor Apa yang Paling Mempengaruhi Nilai?")

if model:
    # 1. Mengambil Koefisien dari Model
    # Pastikan urutan fitur sama persis dengan saat training model!
    feature_names = ['Jam Belajar', 'Kehadiran', 'Kesehatan Mental', 'Jam Tidur', 'Pekerjaan Part-time']
    coefficients = model.coef_[0] if hasattr(model.coef_, '__iter__') and len(model.coef_) == 1 else model.coef_
    
    # Kadang scikit-learn formatnya beda tergantung versi, kita ratakan array-nya
    coefficients = np.array(coefficients).flatten()

    # 2. Membuat DataFrame untuk Visualisasi
    df_importance = pd.DataFrame({
        'Faktor': feature_names,
        'Pengaruh': coefficients
    })

    # Mengurutkan dari pengaruh terbesar
    df_importance = df_importance.sort_values(by='Pengaruh', ascending=True)

    # 3. Visualisasi Bar Chart Horizontal
    fig_imp = go.Figure(go.Bar(
        x=df_importance['Pengaruh'],
        y=df_importance['Faktor'],
        orientation='h',
        marker=dict(color='teal') # Warna batang
    ))

    fig_imp.update_layout(
        title="Tingkat Pengaruh Tiap Faktor terhadap Nilai",
        xaxis_title="Besar Pengaruh (Koefisien)",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig_imp, use_container_width=True)

    # 4. Insight Text Otomatis
    top_factor = df_importance.iloc[-1]['Faktor']
    st.caption(f"ℹ️ **Insight Model:** Berdasarkan data historis, faktor **'{top_factor}'** memiliki pengaruh terbesar dalam menaikkan nilai siswa.")

else:
    st.warning("Visualisasi Feature Importance membutuhkan file model asli.")