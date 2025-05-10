# Laporan Proyek Machine Learning - Imelda Cyntia

## Domain Proyek : Kesehatan

### Latar Belakang

Biaya asuransi kesehatan terus meningkat dari tahun ke tahun, mendorong perusahaan asuransi dan individu untuk mencari pendekatan prediktif yang lebih akurat dalam perencanaan keuangan dan penetapan premi. Berbagai faktor seperti usia, jenis kelamin, indeks massa tubuh (BMI), status merokok, jumlah anak, dan wilayah tempat tinggal diketahui turut memengaruhi besar kecilnya biaya asuransi. Penerapan algoritma machine learning memungkinkan analisis yang lebih presisi terhadap variabel-variabel ini untuk menghasilkan prediksi biaya yang lebih objektif dan berbasis data.

Studi oleh Alhassan dan Batrushi (2022) menunjukkan bahwa penerapan model machine learning secara signifikan dapat meningkatkan akurasi prediksi premi asuransi dibandingkan pendekatan tradisional. Model prediktif ini membantu perusahaan asuransi menentukan premi secara adil dan membantu individu mengelola pengeluaran kesehatannya secara lebih terencana. Selain itu, menurut World Health Organization (2023), pengeluaran kesehatan global meningkat rata-rata 3,9% per tahun sejak tahun 2000, dengan asuransi kesehatan menjadi salah satu komponen pembiayaan utama. Oleh karena itu, membangun model prediksi biaya asuransi berbasis data menjadi langkah penting dalam meningkatkan efisiensi sistem kesehatan secara keseluruhan.

### Referensi

* Alhassan, I., & Batrushi, B. (2022). Health Insurance Premium Prediction Using Machine Learning Techniques. International Journal of Environmental Research and Public Health, 19(13), 7898. [https://doi.org/10.3390/ijerph19137898](https://doi.org/10.3390/ijerph19137898)
* World Health Organization. (2023). Global spending on health: Rising to the pandemic’s challenges. [https://www.who.int/publications/i/item/9789240076643](https://www.who.int/publications/i/item/9789240076643)

## Business Understanding

### Problem Statements

- Bagaimana cara memprediksi biaya asuransi kesehatan berdasarkan fitur-fitur seperti usia, jenis kelamin, kebiasaan merokok, BMI, dan wilayah tempat tinggal?
- Apakah faktor-faktor seperti kebiasaan merokok atau BMI memberikan pengaruh signifikan terhadap besarnya biaya asuransi?

### Goals

- Membangun model prediksi regresi untuk memperkirakan biaya asuransi kesehatan berdasarkan karakteristik pengguna.
- Mengidentifikasi fitur atau atribut yang paling berpengaruh terhadap besarnya biaya asuransi.

### Solution statements

Untuk mencapai tujuan tersebut, pendekatan berikut akan dilakukan:

- Menerapkan beberapa algoritma regresi lainnya seperti K-Nearest Neighbor, Random Forest Regressor dan Gradient Boosting Regressor untuk melihat performa yang lebih baik.

- Menggunakan metrik evaluasi seperti Mean Squared Error (MSE) untuk mengukur akurasi dan efektivitas model.

## Data Understanding

Proyek ini menggunakan dataset Insurance yang tersedia secara publik melalui platform Kaggle: [https://www.kaggle.com/datasets/mirichoi0218/insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance). Dataset ini berisi data demografis dan gaya hidup individu serta jumlah biaya asuransi kesehatan yang dibebankan kepada mereka. Karena targetnya berbentuk nilai numerik (charges), dataset ini cocok digunakan untuk membangun model prediksi regresi.

Dataset terdiri dari 1.338 baris dan 7 kolom, dengan setiap baris mewakili satu individu. Dari eksplorasi awal, tidak ditemukan nilai hilang (missing value), sehingga data dalam kondisi bersih. Selanjutnya, fitur-fitur dibagi berdasarkan tipe datanya:

* Fitur kategorikal: sex, smoker, region
* Fitur numerikal: age, bmi, children, charges

Pengecekan dan penanganan outlier dilakukan khususnya pada fitur numerikal menggunakan metode IQR (Interquartile Range). Data yang mengandung outlier telah dibersihkan agar tidak memengaruhi kualitas prediksi.

### Variabel-variabel pada Insurance Dataset adalah sebagai berikut:

* age: Usia pemegang polis (dalam tahun).
* sex: Jenis kelamin pemegang polis (male atau female).
* bmi: Body Mass Index (indeks massa tubuh).
* children: Jumlah anak yang menjadi tanggungan.
* smoker: Status merokok individu (yes atau no).
* region: Wilayah tempat tinggal di AS (southwest, southeast, northwest, northeast).
* charges: Biaya yang dibebankan oleh asuransi (dalam USD) – ini adalah target prediksi.

### Rubrik/Kriteria Tambahan (Opsional):

Beberapa tahapan eksplorasi data telah dilakukan untuk memahami karakteristik dataset, antara lain:

* Visualisasi univariat menggunakan histogram dan countplot untuk melihat distribusi masing-masing fitur.
* Visualisasi bivariat seperti boxplot dan scatterplot digunakan untuk melihat hubungan antara fitur dan target (charges).
* Heatmap korelasi antar fitur numerik dilakukan untuk mengidentifikasi kekuatan dan arah hubungan antar variabel.
* Pembersihan data dilakukan dengan menghapus outlier berdasarkan boxplot dan metode IQR.

Tahapan-tahapan tersebut bertujuan untuk meningkatkan pemahaman terhadap struktur data dan mendukung pemodelan yang lebih baik di tahap berikutnya.

## Data Preparation

Tahapan data preparation sangat penting untuk memastikan bahwa data yang digunakan dalam proses pemodelan bersih, relevan, dan berada dalam format yang dapat diproses oleh algoritma machine learning. Pada proyek ini, langkah-langkah data preparation dilakukan secara sistematis dan berurutan sebagai berikut:

### 1. Penanganan Outlier

Fitur numerik seperti age, bmi, children, dan charges dianalisis menggunakan boxplot untuk mengidentifikasi outlier.
Outlier kemudian dihapus menggunakan metode IQR (Interquartile Range) agar tidak memengaruhi distribusi data secara ekstrem dan tidak menyebabkan bias dalam proses pelatihan model.

📌 Alasan: Outlier dapat mendistorsi distribusi data dan menurunkan performa model regresi karena model berusaha menyesuaikan terhadap nilai-nilai ekstrem.

### 2. Encoding Fitur Kategorikal

Untuk mengubah fitur kategorikal menjadi numerik:

* sex diubah menjadi 0 untuk female dan 1 untuk male (binary encoding).
* smoker diubah menjadi 0 untuk no dan 1 untuk yes (binary encoding).
* region diubah menggunakan one-hot encoding karena memiliki lebih dari dua kategori (southwest, southeast, northwest, northeast).

📌 Alasan: Model machine learning tidak dapat memproses data bertipe kategorikal secara langsung, sehingga fitur kategorikal perlu diubah menjadi bentuk numerik.

### 3. Pembagian Dataset

Dataset dibagi menjadi data latih dan data uji menggunakan train\_test\_split dari sklearn, dengan rasio 80% data latih dan 20% data uji.

📌 Alasan: Pemisahan ini bertujuan untuk mengevaluasi performa model pada data yang belum pernah dilihat sebelumnya, sehingga dapat menilai kemampuan generalisasi model.

### 4. Standardisasi Fitur Numerik

Fitur numerik age, bmi, dan children distandarisasi menggunakan StandardScaler dari sklearn.

📌 Alasan: Standardisasi diperlukan agar setiap fitur berada dalam skala yang sama. Ini sangat penting untuk model seperti KNN dan Gradient Boosting yang sensitif terhadap perbedaan skala antar fitur.

##################################BATAS SAYA KERJAKAN==================================================
## Evaluation

Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:

- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**:

- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_

- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
