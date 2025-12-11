import streamlit as st
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

model = joblib.load('best_model.pkl')

st.title("Aplikasi Prediksi Nilai Siswa")

study_hours = st.slider("Jam Belajar per Hari", 0.0, 12.0, 2.0)
attendance = st.slider("Persentase Kehadiran (%)", 0.0, 100.0, 80.0)
mental_health = st.slider("Skor Kesehatan Mental (1-10)", 1, 10, 5)
sleep_hours = st.slider("Jam Tidur per Malam", 0.0, 12.0, 7.0)
part_time_job = st.selectbox("Apakah Memiliki Pekerjaan Paruh Waktu?", ("Ya", "Tidak"))

ptj_encode = 1 if part_time_job == "Ya" else 0

if st.button("Prediksi Nilai"):
    input_data = np.array([[study_hours, attendance, mental_health, sleep_hours, ptj_encode]])
    prediction = model.predict(input_data)[0]

    prediction = max(0, min(100, prediction))
    st.success(f"Prediksi Nilai Siswa: {prediction:.2f}")