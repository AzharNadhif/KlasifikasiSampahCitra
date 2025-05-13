from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Path model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "KlasifikasiSampahmodel.h5")

# Load model Keras
model = load_model(MODEL_PATH)

# Daftar kelas sesuai training 
class_names = ['organik', 'anorganik']  # Sesuai folder O dan R

# Fungsi prediksi
def predict_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalisasi
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan batch dimension

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_index]

    return {
        "prediction": class_names[predicted_index],
        "confidence": round(float(confidence) * 100, 2)
    }
