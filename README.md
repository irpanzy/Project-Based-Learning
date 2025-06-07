# Social Media Addiction Prediction using Naive Bayes

## 📋 Deskripsi Proyek

Proyek ini bertujuan untuk memprediksi tingkat kecanduan media sosial pada mahasiswa menggunakan algoritma Gaussian Naive Bayes yang diimplementasikan dari scratch. Dataset yang digunakan adalah **Students Social Media Addiction** yang berisi informasi tentang pola penggunaan media sosial dan dampaknya terhadap mahasiswa.

## 🎯 Tujuan

- Mengklasifikasikan mahasiswa apakah termasuk kategori **kecanduan** atau **tidak kecanduan** media sosial
- Mengimplementasikan algoritma Naive Bayes tanpa menggunakan library yang langsung menyediakan algoritma tersebut
- Menganalisis faktor-faktor yang mempengaruhi kecanduan media sosial

## 📊 Dataset

- Dataset berisi informasi tentang:
- Student_ID: ID unik mahasiswa
- Age: Usia mahasiswa
- Gender: Jenis kelamin
- Academic_Level: Tingkat akademik
- Country: Negara asal
- Avg_Daily_Usage_Hours: Rata-rata jam penggunaan harian
- Most_Used_Platform: Platform yang paling sering digunakan
- Sleep_Hours_Per_Night: Jam tidur per malam
- Mental_Health_Score: Skor kesehatan mental
- Affects_Academic_Performance: Pengaruh terhadap performa akademik
- Relationship_Status: Status hubungan
- Conflicts_Over_Social_Media: Konflik karena media sosial
- Addicted_Score: Skor kecanduan (1-10)

## 🏗️ Struktur Proyek

├── source_code/
│ ├── main.py # File utama untuk menjalankan program
│ ├── preprocessing.py # Modul preprocessing data
│ ├── naive_bayes.py # Implementasi algoritma Naive Bayes
│ └── evaluation.py # Modul evaluasi model
├── data/
│ └── Students Social Media Addiction.csv
└── README.md

## 🔧 Instalasi dan Penggunaan

### 1. Clone Repository

```
git clone https://github.com/irpanzy/social-media-addiction-prediction.git
cd social-media-addiction-prediction
```

### 2. Install Dependencies

```bash
pip install numpy pandas scikit-learn
```

### 3. Menjalankan Program

```bash
cd source_code
python main.py
```

### 4. Output yang Diharapkan

Program akan menampilkan:

```
Akurasi: XX.XX %
Confusion Matrix:
 [[XX XX]
 [XX XX]]
```

## 📝 Penjelasan Setiap Modul

### 1. preprocessing

Modul ini bertanggung jawab untuk mempersiapkan data sebelum digunakan untuk training:

```python
def load_and_preprocess(csv_path):
    # Memuat data dari CSV
    df = pd.read_csv(csv_path)

    # Menghapus kolom Student_ID (tidak diperlukan untuk prediksi)
    df.drop(columns=["Student_ID"], inplace=True)

    # Mengkonversi Addicted_Score menjadi binary classification
    # 1 = kecanduan (score >= 6), 0 = tidak kecanduan (score < 6)
    df["Addicted"] = df["Addicted_Score"].apply(lambda x: 1 if x >= 6 else 0)
    df.drop(columns=["Addicted_Score"], inplace=True)

    # Encoding variabel kategorikal menggunakan LabelEncoder
    categorical_cols = [
        "Gender", "Academic_Level", "Country",
        "Most_Used_Platform", "Affects_Academic_Performance",
        "Relationship_Status"
    ]

    # Normalisasi variabel numerik menggunakan MinMaxScaler
    numerical_cols = [
        "Age", "Avg_Daily_Usage_Hours", "Sleep_Hours_Per_Night",
        "Mental_Health_Score", "Conflicts_Over_Social_Media"
    ]

    # Split data menjadi training dan testing (80:20)
    return train_test_split(X, y, test_size=0.2, random_state=42)
```

### 2. naive_bayes

Implementasi algoritma Gaussian Naive Bayes dari scratch:

```python
class NaiveBayesClassifier:
    def fit(self, X, y):
        # Menghitung parameter statistik untuk setiap kelas
        # - Prior probability: P(class)
        # - Mean dan variance untuk setiap fitur per kelas

    def _gaussian_prob(self, x, mean, var):
        # Menghitung probabilitas menggunakan distribusi Gaussian
        # Formula: (1/√(2πσ²)) * e^(-((x-μ)²)/(2σ²))

    def predict(self, X):
        # Melakukan prediksi menggunakan teorema Bayes
        # Menghitung posterior probability untuk setiap kelas
        # Memilih kelas dengan probabilitas tertinggi
```

#### Algoritma Naive Bayes:

- Training Phase: Menghitung prior probability dan likelihood untuk setiap fitur
- Prediction Phase: Menggunakan teorema Bayes untuk menghitung posterior probability
- Classification: Memilih kelas dengan probabilitas tertinggi

### 3. evaluation

Modul untuk evaluasi performa model:

```python
def evaluate(y_true, y_pred):
    # Menghitung akurasi: (TP + TN) / (TP + TN + FP + FN)
    acc = accuracy_score(y_true, y_pred)

    # Membuat confusion matrix untuk analisis detail
    cm = confusion_matrix(y_true, y_pred)

    return acc, cm
```

### 4. main

File utama yang mengorkestrasi seluruh pipeline:

```python
# 1. Load dan preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess(csv_path)

# 2. Inisialisasi dan training model
model = NaiveBayesClassifier()
model.fit(X_train, y_train)

# 3. Prediksi pada data test
y_pred = model.predict(X_test)

# 4. Evaluasi hasil
print_evaluation(y_test, y_pred)
```

## 📈 Analisis Preprocessing

### 1. Data Cleaning

- Menghapus kolom `Student_ID` karena tidak relevan untuk prediksi
- Mengkonversi `Addicted_Score` menjadi binary classification (threshold = 6)

### 2. Feature Engineering

- Categorical Encoding: Menggunakan LabelEncoder untuk variabel kategorikal
- Normalization: Menggunakan MinMaxScaler untuk variabel numerik agar semua fitur memiliki skala yang sama

### 3. Data Splitting

- 80% data untuk training, 20% untuk testing
- Menggunakan `random_state=42` untuk reprodusibilitas

## 🧠 Desain Algoritma

### Gaussian Naive Bayes

Algoritma ini mengasumsikan bahwa:

- Independence: Setiap fitur independen terhadap fitur lainnya
- Gaussian Distribution: Setiap fitur mengikuti distribusi normal

### Formula Matematika:

```
P(class|features) = P(features|class) × P(class) / P(features)

Dimana:
- P(class|features): Posterior probability
- P(features|class): Likelihood
- P(class): Prior probability
- P(features): Evidence (konstanta)
```

### Implementasi:

- Training: Menghitung mean, variance, dan prior untuk setiap kelas
- Prediction: Menghitung log-probability untuk menghindari numerical underflow
- Smoothing: Menambahkan nilai kecil (1e-6) pada variance untuk mencegah division by zero

## 📊 Model Evaluasi

### Metrik yang Digunakan:

- Accuracy: Persentase prediksi yang benar
- Confusion Matrix: Tabel kontingensi untuk analisis detail

### Confusion Matrix:

```
Predicted
                0    1
Actual    0   [TN] [FP]
          1   [FN] [TP]

Dimana:
- TN: True Negative (benar prediksi tidak kecanduan)
- TP: True Positive (benar prediksi kecanduan)
- FN: False Negative (salah prediksi tidak kecanduan)
- FP: False Positive (salah prediksi kecanduan)
```

## 🎯 Kasus Penggunaan

Proyek ini dapat digunakan untuk:

- Deteksi Dini: Mengidentifikasi mahasiswa yang berisiko kecanduan media sosial
- Intervensi: Memberikan bantuan atau konseling kepada mahasiswa yang teridentifikasi
- Penelitian: Memahami faktor-faktor yang mempengaruhi kecanduan media sosial
- Kebijakan Kampus: Membuat program pencegahan kecanduan media sosial

## 🚀 Pengembangan Lebih Lanjut

Beberapa improvement yang dapat dilakukan:

- Cross-validation untuk evaluasi yang lebih robust
- Feature selection untuk mengurangi dimensi data
- Hyperparameter tuning untuk optimasi performa
- Ensemble methods untuk meningkatkan akurasi
- Visualisasi hasil dan analisis data

## 📄 Lisensi

Proyek ini dibuat untuk tujuan pembelajaran dan penelitian.

## 👥 Kontributor

- Naufal Aflakh Wijayanto - 2211104073
- Irfan Muria - 2211104075
- Frido Afriyanto - 2211104088

---

**Note:** Implementasi ini dibuat dari scratch sesuai dengan requirements tugas, tanpa menggunakan library yang langsung menyediakan algoritma Naive Bayes.
