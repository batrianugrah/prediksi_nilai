from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load Model
try:
    model = joblib.load('gbr_model.joblib')
    print("=== Model berhasil dimuat! ===")
except Exception as e:
    print(f"=== ERROR Load Model: {e} ===")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Terima data JSON
        payload = request.get_json(force=True)
        
        # 2. Cek apakah ada key 'data' (sesuai format JSON Anda)
        if 'data' not in payload:
            return jsonify({'error': "Format JSON harus memiliki key 'data'"}), 400
            
        # 3. Ambil list angka di dalamnya
        # payload['data'] isinya adalah [[2.58..., -9.17..., ...]]
        input_values = payload['data']
        
        # 4. Lakukan Prediksi
        # Kita kirim list langsung (array) agar model tidak bingung mencari nama kolom
        prediction = model.predict(input_values)
        
        return jsonify({'prediction': prediction[0]})

    except Exception as e:
        # Tampilkan error lengkap jika terjadi sesuatu
        print(f"Error saat prediksi: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Gunakan setting ini agar stabil di Windows/CMD
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)