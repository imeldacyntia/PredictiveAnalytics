# Laporan Proyek Machine Learning - Imelda Cyntia

## Domain Proyek : Kesehatan

### Latar Belakang

Dalam beberapa tahun terakhir, biaya asuransi kesehatan mengalami peningkatan yang signifikan, mendorong perusahaan asuransi dan individu untuk mencari solusi prediktif yang lebih akurat dalam perencanaan keuangan. Dengan banyaknya faktor yang memengaruhi biaya asuransi seperti usia, jenis kelamin, indeks massa tubuh (BMI), status merokok, dan wilayah tempat tinggal, penerapan machine learning dapat menjadi pendekatan efektif untuk memprediksi biaya asuransi secara lebih presisi. Model prediktif berbasis data memungkinkan perusahaan untuk menentukan premi secara lebih adil dan membantu individu merencanakan pengeluaran kesehatannya dengan lebih baik (Tajaddodi Nodehi et al., 2023).

Menurut laporan World Health Organization (2023), pengeluaran kesehatan global meningkat rata-rata 3,9% per tahun sejak 2000, dengan pembiayaan pribadi dan asuransi sebagai dua komponen utama. Sistem prediksi berbasis machine learning berpotensi mempercepat proses analisis dan memberikan estimasi biaya yang lebih cepat dan akurat. Oleh karena itu, membangun model regresi untuk memprediksi biaya asuransi kesehatan berdasarkan data demografis dan gaya hidup menjadi penting dalam mendukung efisiensi sistem kesehatan secara keseluruhan.

### Referensi

- Tajaddodi Nodehi, M., Hosseini Khatibani, S., Yazdinejad, M., & Zolfi, S. (2023). Predicting people's health insurance costs using machine learning and ensemble learning methods. Iranian Journal of Insurance Research, 13(1), 1-14. [https://doi.org/10.22056/ijir.2024.01.01](https://doi.org/10.22056/ijir.2024.01.01)
- World Health Organization. (2023). _Global spending on health: Rising to the pandemic’s challenges_. [https://www.who.int/publications/i/item/9789240076643](https://www.who.int/publications/i/item/9789240076643)

## Business Understanding

### Problem Statements

- Bagaimana cara memprediksi biaya asuransi kesehatan berdasarkan fitur-fitur seperti usia, jenis kelamin, kebiasaan merokok, BMI, dan wilayah tempat tinggal?
- Apakah faktor-faktor seperti kebiasaan merokok atau BMI memberikan pengaruh signifikan terhadap besarnya biaya asuransi?

### Goals

- Membangun model prediksi regresi untuk memperkirakan biaya asuransi kesehatan berdasarkan karakteristik pengguna.
- Mengidentifikasi fitur atau atribut yang paling berpengaruh terhadap besarnya biaya asuransi.

### Solution statements

Untuk mencapai tujuan tersebut, pendekatan berikut akan dilakukan:

- Mengembangkan baseline model regresi linier (Linear Regression) sebagai pembanding awal.

- Menerapkan beberapa algoritma regresi lainnya seperti Random Forest Regressor dan Gradient Boosting Regressor untuk melihat performa yang lebih baik.

- Menggunakan metrik evaluasi seperti Mean Absolute Error (MAE), Mean Squared Error (MSE), dan R-squared (R²) untuk mengukur akurasi dan efektivitas model.

## Data Understanding

Proyek ini menggunakan dataset _Insurance_ yang tersedia secara publik melalui platform [Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance). Dataset ini berisi informasi mengenai data demografis dan gaya hidup seseorang serta biaya asuransi kesehatan yang dibayarkan. Data ini sangat cocok digunakan untuk membangun model prediksi regresi karena target variabelnya adalah nilai numerik (biaya asuransi), dan tersedia berbagai fitur yang dapat memengaruhi biaya tersebut.

Dataset ini terdiri dari **1.338 baris data** dan **7 kolom**. Setiap baris merepresentasikan satu individu dengan atribut-atribut terkait kesehatan dan demografi.

### Variabel-variabel pada _Insurance Dataset_ adalah sebagai berikut:

- **age**: Usia dari pemegang polis asuransi (dalam tahun).
- **sex**: Jenis kelamin dari pemegang polis (`male` atau `female`).
- **bmi**: _Body Mass Index_ (Indeks Massa Tubuh) dari individu.
- **children**: Jumlah tanggungan anak yang dimiliki.
- **smoker**: Status merokok individu (`yes` atau `no`).
- **region**: Wilayah tempat tinggal individu di Amerika Serikat (`southwest`, `southeast`, `northwest`, atau `northeast`).
- **charges**: Biaya yang ditagihkan oleh asuransi (dalam satuan dolar AS)

### Exploratory Data Analysis (EDA)

Untuk memahami data lebih dalam, dilakukan beberapa langkah eksplorasi awal, antara lain:

1. **Distribusi Target (`charges`)**
   Distribusi `charges` bersifat _right-skewed_ (miring ke kanan), menunjukkan bahwa sebagian besar individu membayar premi yang relatif rendah, tetapi ada beberapa outlier dengan biaya sangat tinggi.

2. **Hubungan antara Fitur dan Target**

   - **smoker** memiliki pengaruh sangat besar terhadap biaya asuransi. Rata-rata biaya untuk perokok jauh lebih tinggi dibandingkan non-perokok.
   - **bmi** dan **charges** menunjukkan hubungan yang lebih kuat pada kelompok perokok, terutama bagi yang mengalami obesitas.
   - **age** juga menunjukkan korelasi positif terhadap `charges`: semakin tua seseorang, semakin besar kemungkinan biaya asuransinya.

3. **Visualisasi Data**
   Beberapa teknik visualisasi yang digunakan antara lain:

   - _Histogram_ untuk distribusi fitur numerik (`age`, `bmi`, `charges`)
   - _Boxplot_ untuk melihat perbandingan `charges` berdasarkan `smoker`, `sex`, dan `region`
   - _Pairplot_ dan _correlation heatmap_ untuk melihat korelasi antar fitur numerik

Visualisasi ini membantu memahami pola-pola dalam data dan memberikan wawasan awal tentang fitur mana yang kemungkinan besar berkontribusi terhadap prediksi biaya asuransi.

########################YANG SAYA UBAH####################################

## Data Preparation

Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**:

- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling

Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**:

- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

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
