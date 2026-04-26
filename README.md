# Dokumentasi Proyek Klasifikasi Kucing dan Anjing Menggunakan CNN

## 📋 Overview Proyek

Proyek ini bertujuan untuk membangun model **Convolutional Neural Network (CNN)** untuk klasifikasi gambar kucing dan anjing. Notebook ini menggunakan TensorFlow dan Keras untuk implementasi deep learning, dengan setiap langkah dijelaskan secara detail agar mudah dipahami oleh pemula maupun praktisi machine learning.

---

## 📊 Dataset

### Sumber Dataset

Dataset yang digunakan dalam proyek ini berasal dari **Kaggle**:

- **Nama Dataset**: Cat or Dog Image Classification
- **Link**: [Kaggle Dataset](https://www.kaggle.com/datasets/sunilthite/cat-or-dog-image-classification)
- **Author**: sunilthite

### Struktur Dataset

```
cat_dog_dataset/
├── Train/
│   ├── Cat/      # Gambar kucing untuk training
│   └── Dog/     # Gambar anjing untuk training
└── Test/
    ├── Cat/      # Gambar kucing untuk testing
    └── Dog/     # Gambar anjing untuk testing
```

---

## 🛠️ Teknologi dan Library

| Library | Fungsi |
|---------|--------|
| TensorFlow | Framework deep learning untuk membangun dan melatih model CNN |
| NumPy | Manipulasi array dan operasi numerik |
| Matplotlib | Visualisasi data dan plotting gambar |
| Keras (TensorFlow) | High-level API untuk membangun neural network |
| OpenCV (cv2) | Pemrosesan gambar (opsional) |
| Scikit-learn | Evaluasi model (classification report, confusion matrix) |
| Seaborn | Visualisasi heatmap untuk confusion matrix |

---

## 📝 Penjelasan Setiap Cell Notebook

### Cell 1: Judul dan Deskripsi Proyek

```markdown
# Classification Cat and Dog using CNN

Notebook ini membahas proses klasifikasi gambar kucing dan anjing menggunakan 
Convolutional Neural Network (CNN) dengan TensorFlow dan Keras. Setiap langkah 
akan dijelaskan secara detail agar mudah dipahami.
```

**Penjelasan**: Cell ini merupakan pengantar proyek yang menjelaskan tujuan utama notebook, yaitu membangun model CNN untuk klasifikasi gambar kucing dan anjing.

---

### Cell 2-3: Import Library

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image_dataset_from_directory
```

**Penjelasan**:
- `tensorflow`: Framework utama untuk deep learning
- `numpy`: Untuk operasi array dan manipulasi data numerik
- `matplotlib.pyplot`: Untuk visualisasi grafik dan gambar
- `image_dataset_from_directory`: Fungsi untuk memuat dataset gambar langsung dari folder

---

### Cell 4-5: Inisialisasi Direktori

```python
dir = "D:\All Projects\CNN Basic"
```

**Penjelasan**: Menentukan path direktori utama proyek untuk memudahkan akses ke dataset dan folder penyimpanan model.

---

### Cell 6-9: Import Dataset dari Kaggle

```python
import kagglehub

path = kagglehub.dataset_download("sunilthite/cat-or-dog-image-classification")
print("Path to dataset files:", path)
```

**Penjelasan**:
- Menggunakan library `kagglehub` untuk mengunduh dataset langsung dari Kaggle
- Dataset otomatis diunduh ke folder cache dan path-nya disimpan dalam variabel `path`

---

### Cell 10-11: Memindahkan Dataset ke Folder Project

```python
import os
import shutil

src = path
dst = f"{dir}\\cat_dog_dataset"

if not os.path.exists(dst):
    shutil.copytree(src, dst)
    print(f"Dataset berhasil disalin ke {dst}")
else:
    print(f"Folder tujuan {dst} sudah ada.")
```

**Penjelasan**:
- Menyalin dataset dari folder unduhan ke folder proyek
- Memeriksa apakah folder sudah ada untuk menghindari duplikasi

---

### Cell 12-13: Alternatif Google Drive (Commented)

```python
# from google.colab import drive
# drive.mount('/content/drive')
```

**Penjelasan**: Alternatif untuk pengguna Google Colab yang ingin memuat dataset dari Google Drive.

---

### Cell 14-15: Pre-Processing Dataset Pelatihan

```python
train_dataset = image_dataset_from_directory(
    dir + "/cat_dog_dataset/Train",
    image_size=(150, 150),
    batch_size=32,
    label_mode='binary'
)

train_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.Rescaling(1./255)
])

train_dataset = train_dataset.map(
    lambda x, y: (train_augmentation(x, training=True), y)
)
```

**Penjelasan**:
- `image_dataset_from_directory`: Memuat gambar dari folder dengan struktur label otomatis
- `image_size=(150, 150)`: Mengubah ukuran semua gambar menjadi 150x150 piksel
- `batch_size=32`: Jumlah gambar yang diproses dalam satu iterasi pelatihan
- `label_mode='binary'`: Label 0 untuk kucing, 1 untuk anjing

**Data Augmentation**:
- `RandomFlip('horizontal')`: Membalik gambar secara horizontal untuk menambah variasi
- `RandomRotation(0.1)`: Memutar gambar secara acak hingga 10%
- `RandomZoom(0.1)`: Melakukan zoom acak hingga 10%
- `Rescaling(1./255)`: Normalisasi piksel ke rentang [0, 1]

---

### Cell 16-17: Pre-Processing Dataset Pengujian

```python
test_dataset = image_dataset_from_directory(
    dir + "/cat_dog_dataset/Test",
    image_size=(150, 150),
    batch_size=32,
    label_mode='binary'
)

test_augmentation = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255)
])

test_dataset = test_dataset.map(
    lambda x, y: (test_augmentation(x, training=False), y)
)
```

**Penjelasan**:
- Dataset pengujian hanya dinormalisasi (tanpa augmentasi) untuk menjaga konsistensi evaluasi
- Augmentasi hanya diterapkan pada data pelatihan untuk mencegah data leakage

---

### Cell 18-19: Inisialisasi Model CNN

```python
cnn = tf.keras.Sequential()
```

**Penjelasan**: Membuat objek model Sequential dari Keras untuk membangun arsitektur CNN secara berurutan.

---

### Cell 20-21: Convolutional Layer 1

```python
cnn.add(tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=(3, 3),
    activation='relu',
    input_shape=(150, 150, 3)
))
```

**Penjelasan**:
- `Conv2D`: Layer konvolusi untuk mengekstrak fitur dari gambar
- `filters=32`: Jumlah filter/kernel yang digunakan (mengekstrak 32 fitur berbeda)
- `kernel_size=(3, 3)`: Ukuran filter 3x3 piksel
- `activation='relu'`: Fungsi aktivasi ReLU untuk menambahkan non-linearitas
- `input_shape=(150, 150, 3)`: Ukuran input (tinggi, lebar, channel warna RGB)

---

### Cell 22-23: Pooling Layer 1

```python
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
```

**Penjelasan**:
- `MaxPool2D`: Layer pooling untuk mengurangi dimensi spatial
- `pool_size=2`: Window pooling 2x2
- `strides=2`: Pergeseran window pooling (tanpa overlap)
- Berfungsi untuk mengurangi computational load dan mencegah overfitting

---

### Cell 24-25: Convolutional Layer 2 dan Pooling Layer 2

```python
cnn.add(tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=(3, 3),
    activation='relu',
))

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
```

**Penjelasan**:
- Layer konvolusi kedua untuk mengekstrak fitur yang lebih kompleks
- Tidak perlu mendefinisikan input_shape karena otomatis terhubung dengan layer sebelumnya

---

### Cell 26-27: Flatten Layer

```python
cnn.add(tf.keras.layers.Flatten())
```

**Penjelasan**: Meratakan output dari layer konvolusi menjadi vektor 1 dimensi untuk masuk ke fully connected layer.

---

### Cell 28-29: Fully Connected Layer

```python
cnn.add(tf.keras.layers.Dense(256, activation='relu'))
```

**Penjelasan**:
- `Dense(256)`: Layer fully connected dengan 256 neuron
- `activation='relu'`: Fungsi aktivasi ReLU untuk pembelajaran non-linear

---

### Cell 30-31: Output Layer

```python
cnn.add(tf.keras.layers.Dense(1, activation='sigmoid'))
```

**Penjelasan**:
- `Dense(1)`: 1 neuron output untuk klasifikasi biner
- `activation='sigmoid'`: Menghasilkan probabilitas antara 0 dan 1
  - Nilai < 0.5: Kucing
  - Nilai >= 0.5: Anjing

---

### Cell 32-33: Kompilasi Model

```python
cnn.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

**Penjelasan**:
- `optimizer='adam'`: Optimizer yang efisien untuk gradient descent
- `loss='binary_crossentropy'`: Fungsi loss untuk klasifikasi biner
- `metrics=['accuracy']`: Metrik evaluasi selama pelatihan

---

### Cell 34-35: Pelatihan Model

```python
cnn.fit(x=train_dataset, validation_data=test_dataset, epochs=25)
```

**Penjelasan**:
- `train_dataset`: Data pelatihan
- `validation_data=test_dataset`: Data validasi (menggunakan test set)
- `epochs=25`: Jumlah iterasi penuh melalui seluruh dataset

---

### Cell 36-37: Penyimpanan Model

```python
import os

os.makedirs('saved_models', exist_ok=True)

cnn.save('saved_models/cat_dog_model.keras')
cnn.save('saved_models/cat_dog_model.h5')

loaded_model = tf.keras.models.load_model('saved_models/cat_dog_model.keras')
```

**Penjelasan**:
- Menyimpan model dalam dua format: `.keras` (format baru) dan `.h5` (format HDF5)
- `loaded_model`: Memuat model yang tersimpan untuk inferensi

---

### Cell 38-39: Prediksi Gambar Tunggal

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

test_image_path = dir + r"\cat_dog_dataset\Test\Cat\cat.16.jpg"
test_image = image.load_img(test_image_path, target_size=(150, 150))
test_image_array = image.img_to_array(test_image)
test_image_array = np.expand_dims(test_image_array, axis=0)
test_image_array = test_image_array / 255.0

prediction = cnn.predict(test_image_array)
if prediction[0][0] != 1:
    predicted_class = 'Kucing'
else:
    predicted_class = 'Anjing'

plt.imshow(test_image)
plt.title(f"Prediksi: {predicted_class}")
plt.axis('off')
plt.show()
```

**Penjelasan**:
- Memuat gambar uji dan mengubah ukurannya menjadi 150x150 piksel
- Mengonversi gambar ke array NumPy dan menambahkan dimensi batch
- Normalisasi piksel ke rentang [0, 1]
- Melakukan prediksi dan menampilkan hasil

---

### Cell 40-41: Evaluasi Model

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_true = []
y_pred = []

for images, labels in test_dataset:
    predictions = cnn.predict(images)
    predicted_labels = (predictions > 0.5).astype(int).flatten()
    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)

print("Classification Report:")
print(classification_report(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

if not os.path.exists("images"):
    os.makedirs("images")
    plt.savefig("images/confusion_matrix.png")

plt.show()
```

**Penjelasan**:
- `classification_report`: Menampilkan precision, recall, dan F1-score untuk setiap kelas
- `confusion_matrix`: Matriks yang menunjukkan jumlah prediksi benar dan salah
- Menyimpan visualisasi confusion matrix ke folder `images`

---

## 🏗️ Arsitektur Model CNN

```
┌─────────────────────────────────────────────────────────────┐
│                    ARSITEKTUR MODEL                         │
├─────────────────────────────────────────────────────────────┤
│  Input Image (150, 150, 3)                                 │
│         ↓                                                   │
│  Conv2D (32 filters, 3x3) + ReLU                           │
│         ↓                                                   │
│  MaxPool2D (2x2)                                           │
│         ↓                                                   │
│  Conv2D (32 filters, 3x3) + ReLU                           │
│         ↓                                                   │
│  MaxPool2D (2x2)                                           │
│         ↓                                                   │
│  Flatten                                                    │
│         ↓                                                   │
│  Dense (256 neurons) + ReLU                                │
│         ↓                                                   │
│  Dense (1 neuron) + Sigmoid                                 │
│         ↓                                                   │
│  Output (Probabilitas: 0=Kucing, 1=Anjing)                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 📈 Hyperparameter yang Digunakan

| Parameter | Nilai | Keterangan |
|------------|-------|------------|
| Image Size | 150x150 | Ukuran gambar input |
| Batch Size | 32 | Jumlah gambar per iterasi |
| Epochs | 25 | Jumlah iterasi pelatihan |
| Optimizer | Adam | Adaptive Moment Estimation |
| Loss Function | Binary Crossentropy | Untuk klasifikasi biner |
| Conv2D Filters | 32 (setiap layer) | Jumlah filter konvolusi |
| Dense Units | 256 | Neuron pada fully connected layer |

---

## 📊 Hasil dan Evaluasi

Model ini menghasilkan:
- **Accuracy**: Persentase prediksi yang benar
- **Precision**: Rasio prediksi positif yang benar
- **Recall**: Kemampuan mendeteksi kelas positif
- **F1-Score**: Harmonic mean dari precision dan recall

Confusion matrix menunjukkan:
- True Positive: Jumlah prediksi benar untuk Anjing
- True Negative: Jumlah prediksi benar untuk Kucing
- False Positive: Prediksi Anjing yang sebenarnya Kucing
- False Negative: Prediksi Kucing yang sebenarnya Anjing

---

## 💾 Penyimpanan Model

Model yang telah dilatih disimpan dalam folder `saved_models/` dengan dua format:
1. `cat_dog_model.keras` - Format native Keras
2. `cat_dog_model.h5` - Format HDF5

---

## 🙏 Credit dan Acknowledgments

### Dataset

- **Sumber**: [Kaggle - Cat or Dog Image Classification](https://www.kaggle.com/datasets/sunilthite/cat-or-dog-image-classification)
- **Author**: sunilthite
- **Lisensi**: Silakan lihat halaman Kaggle untuk informasi lisensi lengkap

### Library dan Framework

- **TensorFlow/Keras**: [https://www.tensorflow.org/](https://www.tensorflow.org/)
- **NumPy**: [https://numpy.org/](https://numpy.org/)
- **Matplotlib**: [https://matplotlib.org/](https://matplotlib.org/)
- **Scikit-learn**: [https://scikit-learn.org/](https://scikit-learn.org/)

---

## 📚 Referensi Tambahan

1. [TensorFlow CNN Tutorial](https://www.tensorflow.org/tutorials/images/cnn)
2. [Keras Sequential API](https://keras.io/api/models/sequential/)
3. [Data Augmentation](https://keras.io/api/data_augmentation/)

---

*Dokumen ini dibuat secara otomatis dari notebook CNN.ipynb*