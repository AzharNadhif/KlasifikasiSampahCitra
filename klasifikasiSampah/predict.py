import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import os

# Ambil path model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "klasifikasiSampah.pth")

# Buat ulang struktur model dengan output 6 kelas (sesuai training)
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 6)  # Sesuaikan dengan jumlah kelas saat training

# Load state_dict ke model
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()  # Set model ke mode evaluasi

# Transformasi gambar (sesuai dengan training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Mapping kelas dari 6 kategori ke 2 kategori
category_mapping = {
    "cardboard": "organik",
    "paper": "organik",
    "trash": "organik",
    "glass": "anorganik",
    "metal": "anorganik",
    "plastic": "anorganik"
}

# Daftar kelas sesuai dengan training
original_class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

def predict_image(image):
    image = transform(image).unsqueeze(0)  # Tambah batch dimensi
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)  # Konversi ke probabilitas
        confidence, predicted = torch.max(probabilities, 0)  # Ambil prediksi dengan confidence tertinggi
    
    predicted_label = original_class_names[predicted.item()]
    predicted_category = category_mapping[predicted_label]

    return {
        "prediction": predicted_category,
        "confidence": round(confidence.item() * 100, 2)  # Ubah ke persentase
    }
