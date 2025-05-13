import requests

# URL backend Flask
url = "http://127.0.0.1:5000/predict"

# Path gambar yang akan diuji (pastikan path sesuai dengan lokasi file Anda)
file_path = r"D:\PunyaNadip\Coolyeah\SEMESTER 6\COMVIS\kelomfuck\gambaruji.png"

# Buka gambar untuk dikirim ke backend
files = {"file": open(file_path, "rb")}

# Kirim request ke backend
response = requests.post(url, files=files)

# Tampilkan hasil prediksi
print(response.json())
