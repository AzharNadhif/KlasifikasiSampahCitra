from flask import Flask, request, jsonify
from flask_cors import CORS  # Tambahkan ini
from PIL import Image
import torch
import io
from model.predict import predict_image  # Import fungsi prediksi

app = Flask(__name__)
CORS(app)  # Mengaktifkan CORS agar frontend bisa mengakses API Flask

@app.route('/')
def home():
    return "API Klasifikasi Sampah Aktif!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    
    # Prediksi klasifikasi gambar
    result = predict_image(image)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
