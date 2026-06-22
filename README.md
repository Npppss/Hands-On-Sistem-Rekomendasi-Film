# Sistem Rekomendasi Film Berbasis Machine Learning & Deep Learning (MLOps)

Proyek ini bertujuan untuk membangun sistem rekomendasi film menggunakan tiga pendekatan populer: **Demographic Filtering**, **Content-Based Filtering**, dan **Collaborative Filtering** berbasis neural network. Selain itu, proyek ini dirancang dengan pendekatan MLOps menggunakan **DVC (Data Version Control)** dan **Dagshub** untuk melacak versi data serta merekam hasil eksperimen model secara otomatis.

---

## 1. Problem Statement

Di era digital saat ini, jumlah konten film yang tersedia di platform streaming sangat melimpah. Hal ini sering kali menimbulkan fenomena *choice overload*, di mana pengguna kesulitan memilih film yang sesuai dengan minat mereka. 

Untuk meningkatkan pengalaman pengguna dan retensi platform, sistem rekomendasi otomatis sangat diperlukan. Sistem ini bertugas menganalisis karakteristik film dan pola interaksi pengguna (rating) untuk menyodorkan rekomendasi film yang relevan dan terpersonalisasi.

---

## 2. Tujuan Proyek

1. **Demographic Filtering**: Menyajikan daftar film terpopuler secara umum dengan rating rata-rata terbaik menggunakan perhitungan statistik yang adil.
2. **Content-Based Filtering**: Merekomendasikan film serupa berdasarkan kemiripan atribut konten (genre, sutradara, aktor utama, kata kunci tema, dan overview cerita).
3. **Collaborative Filtering**: Memprediksi preferensi pribadi pengguna terhadap film yang belum ditonton dengan melatih representasi laten user-item menggunakan neural network.
4. **MLOps Integration**: Menggunakan DVC untuk mengelola versi dataset berukuran besar tanpa membebani repositori Git, serta mengintegrasikan Dagshub MLflow untuk melacak parameter dan loss pelatihan model secara remote.

---

## 3. Dataset

Dataset yang digunakan bersumber dari Kaggle/GroupLens yang mencakup informasi film dan interaksi user:

| Nama File | Deskripsi | Ukuran |
|---|---|---|
| `movies_metadata.csv` | Atribut utama film (judul, genre, rata-rata rating, budget, dll.) | ~34 MB |
| `credits.csv` | Daftar pemeran utama (cast) dan kru di balik layar (crew) | ~190 MB |
| `keywords.csv` | Kata kunci atau tag pencarian yang mendeskripsikan tema film | ~6.2 MB |
| `ratings_small.csv` | Subset rating pengguna (~100.000 rating dari 700 user pada 9.000 film) | ~2.4 MB |

*Catatan: File dataset mentah berukuran besar tidak diunggah langsung ke GitHub, melainkan dilacak menggunakan DVC dan disimpan di cloud remote storage milik Dagshub.*

---

## 4. Metode & Alur Sistem

### A. Demographic Filtering (IMDB Weighted Rating)
Menggunakan rumus **Weighted Rating IMDB** untuk memastikan penilaian yang adil bagi film dengan ribuan vote dibandingkan dengan film rating tinggi yang hanya memiliki sedikit vote:
$$W = \frac{v}{v+m} \times R + \frac{m}{v+m} \times C$$

Di mana:
* $v$ = Jumlah vote film (`vote_count`)
* $m$ = Ambang batas minimum vote (menggunakan persentil ke-90)
* $R$ = Rata-rata rating film (`vote_average`)
* $C$ = Rata-rata rating seluruh dataset film

### B. Content-Based Filtering
* **Ekstraksi Fitur**: Mengambil sutradara, 3 aktor utama, genre, kata kunci, dan overview, lalu membersihkan spasi (contoh: "Johnny Depp" $\rightarrow$ "johnnydepp").
* **Metadata Soup**: Menggabungkan seluruh teks fitur menjadi satu string deskripsi.
* **TF-IDF & Cosine Similarity**: Mengubah string tersebut menjadi matriks vektor TF-IDF, lalu mengukur kemiripan kosinus antar film. Untuk mencegah *Out Of Memory* (OOM), pencarian dibatasi pada top 15.000 film paling populer.

### C. Collaborative Filtering (Neural RecommenderNet)
* Menerapkan neural network berbasis **Keras Embedding**.
* Memetakan indeks user dan film ke ruang laten berdimensi rendah (`embedding_size=50`).
* Menghitung dot product antara user vector dan movie vector, lalu ditambahkan bias user dan bias movie.
* Fungsi aktivasi Sigmoid digunakan untuk menghasilkan estimasi rating ternormalisasi [0, 1] yang kemudian diskalakan kembali ke skala rating asli (1-5).

---

## 5. Implementasi MLOps (DVC & Dagshub)

Proyek ini menerapkan prinsip MLOps dengan menggunakan:
1. **DVC (Data Version Control)**:
   * Menginisialisasi DVC pada direktori kerja untuk melacak dataset besar di folder `data/`.
   * Menghubungkan DVC dengan remote storage gratis yang disediakan oleh Dagshub.
   * File `.csv` yang besar diabaikan oleh Git (tercantum di `.gitignore`) dan versi datanya dikunci di dalam file metadata `.dvc`.
2. **Dagshub MLflow**:
   * Melacak eksperimen model deep learning secara remote.
   * Menyimpan metrik evaluasi model (Training & Validation Loss) dan parameter model (`embedding_size`, `learning_rate`, dll.) secara otomatis selama pelatihan menggunakan callback `mlflow.tensorflow.autolog()`.

---

## 6. Cara Menjalankan Proyek

### 1. Persiapan Environment
Pastikan Anda menggunakan Python 3.10+ (atau environment anaconda) dan instal seluruh pustaka yang dibutuhkan:
```bash
pip install -r requirements.txt
```

### 2. Konfigurasi DVC & Tarik Data
Jika repositori ditarik dari Git dan data lokal belum tersedia, konfigurasikan remote DVC ke Dagshub dan lakukan pull:
```bash
dvc remote add -d origin https://dagshub.com/<username>/<repo_name>.dvc
dvc pull -r origin
```

### 3. Eksekusi Jupyter Notebook
Jalankan Jupyter Notebook untuk melihat proses data preprocessing, visualisasi chart, pelatihan neural network, dan hasil rekomendasi film:
```bash
jupyter notebook model.ipynb
```

---

## 7. Output Proyek

Setelah notebook dieksekusi, sistem akan menghasilkan beberapa output visual:
1. **Grafik Top 10 Film Terpopuler** berdasarkan skor weighted rating IMDB secara umum.
2. **Rekomendasi Content-Based**: Rekomendasi 10 film serupa berdasarkan kemiripan deskripsi dan kru.
3. **Kurva Loss Training & Validation** yang menunjukkan tren konvergensi model deep learning.
4. **Rekomendasi Personal**: Rekomendasi 10 film teratas yang disesuaikan secara khusus bagi user tertentu (misalnya User ID 5) yang belum pernah ia tonton sebelumnya.

---

## Author
* **Project**: Sistem Rekomendasi Film Berbasis Machine Learning & Deep Learning (MLOps)
* **Status**: Selesai & Terverifikasi
