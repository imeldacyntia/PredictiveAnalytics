# Laporan Proyek Machine Learning - Imelda Cyntia

![Rating](img/rating.png)

## Domain Proyek : Kesehatan

### Latar Belakang

Biaya asuransi kesehatan terus meningkat dari tahun ke tahun, mendorong perusahaan asuransi dan individu untuk mencari pendekatan prediktif yang lebih akurat dalam perencanaan keuangan dan penetapan premi. Berbagai faktor seperti usia, jenis kelamin, indeks massa tubuh (BMI), status merokok, jumlah anak, dan wilayah tempat tinggal diketahui turut memengaruhi besar kecilnya biaya asuransi. Penerapan algoritma machine learning memungkinkan analisis yang lebih presisi terhadap variabel-variabel ini untuk menghasilkan prediksi biaya yang lebih objektif dan berbasis data.

Studi oleh Alhassan dan Batrushi (2022) menunjukkan bahwa penerapan model machine learning secara signifikan dapat meningkatkan akurasi prediksi premi asuransi dibandingkan pendekatan tradisional. Model prediktif ini membantu perusahaan asuransi menentukan premi secara adil dan membantu individu mengelola pengeluaran kesehatannya secara lebih terencana. Selain itu, menurut World Health Organization (2023), pengeluaran kesehatan global meningkat rata-rata 3,9% per tahun sejak tahun 2000, dengan asuransi kesehatan menjadi salah satu komponen pembiayaan utama. Oleh karena itu, membangun model prediksi biaya asuransi berbasis data menjadi langkah penting dalam meningkatkan efisiensi sistem kesehatan secara keseluruhan.

### Referensi

* Alhassan, I., & Batrushi, B. (2022). Health Insurance Premium Prediction Using Machine Learning Techniques. International Journal of Environmental Research and Public Health, 19(13), 7898. [https://doi.org/10.3390/ijerph19137898](https://doi.org/10.3390/ijerph19137898)
* World Health Organization. (2023). Global spending on health: Rising to the pandemic’s challenges. [https://www.who.int](https://www.who.int)

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

Beberapa tahapan eksplorasi data telah dilakukan untuk memahami karakteristik dataset, antara lain:

  #### 1. Menggunakan visualisasi boxplot untuk melihat apakah terdapat outliers.

![Outliner Sebelum](img/outliner_sebelum.png)  
Dari gambar di atas menunjukkan bahwa beberapa fitur numerik yang dianalisis mengandung nilai pencilan (outliers).

  #### 2. Visualisasi univariat menggunakan countplot dan histogram untuk melihat distribusi masing-masing fitur.

![UnivariateAnalysis1](img/UnivariateAnalysis1.png)  
Fitur kategorikal menunjukkan beberapa pola penting. Distribusi sex antara laki-laki dan perempuan relatif seimbang, sehingga model tidak menunjukkan potensi bias terhadap jenis kelamin tertentu. Pada fitur smoker, mayoritas responden merupakan non-smoker, namun proporsi perokok tetap signifikan dan menjadi variabel penting karena memiliki pengaruh besar terhadap biaya asuransi. Sementara itu, distribusi region cukup merata di antara keempat wilayah, dengan sedikit dominasi pada wilayah southeast, yang tetap memberikan representasi geografis yang adil dalam dataset.

![UnivariateAnalysis2](img/UnivariateAnalysis2.png)

Fitur numerikal juga memberikan insight yang berharga. Distribusi usia (age) cukup merata dengan konsentrasi pada usia dewasa muda hingga pertengahan. BMI terdistribusi hampir normal dengan sedikit skew ke kanan, dan setelah penghapusan outlier, distribusinya terlihat lebih bersih meskipun masih terdapat beberapa individu dengan nilai BMI tinggi. Fitur children menunjukkan bahwa sebagian besar individu memiliki 0 hingga 2 anak, tanpa adanya nilai ekstrem. Terakhir, fitur charges menunjukkan distribusi yang sangat skew ke kanan, menandakan bahwa hanya sebagian kecil individu yang memiliki biaya asuransi sangat tinggi, yang kemungkinan besar berkaitan dengan status merokok atau kondisi kesehatan tertentu.

  #### 3. Visualisasi bivariat seperti boxplot dan scatterplot digunakan untuk melihat hubungan antara fitur dan target (charges).

![BivariateAnalysis1](img/BivariateAnalysis1.png)  
Pada variabel sex, rata-rata biaya asuransi pria sedikit lebih tinggi dibanding wanita, namun perbedaannya tidak signifikan dan distribusinya cukup mirip. Variabel smoker menunjukkan pengaruh paling signifikan, di mana perokok memiliki biaya asuransi yang jauh lebih tinggi dibanding non-perokok, dan hampir semua outlier biaya tinggi berasal dari kelompok perokok. Sementara itu, variabel region menunjukkan bahwa distribusi charges relatif serupa di setiap wilayah, tanpa perbedaan signifikan antar region.

![BivariateAnalysis2](img/BivariateAnalysis2.png)

Terdapat korelasi positif antara usia dan charges yaitu semakin tua usia individu, semakin tinggi biaya asuransi yang dikenakan, kemungkinan karena peningkatan risiko kesehatan. Sementara itu, BMI tidak menunjukkan hubungan linier yang kuat dengan charges, namun terdapat beberapa outlier pada BMI tinggi yang menunjukkan potensi biaya lebih besar akibat kondisi seperti obesitas. Untuk variabel children, tidak ditemukan pengaruh signifikan terhadap charges, karena distribusi biaya relatif stabil terlepas dari jumlah anak.

  #### 4. Heatmap korelasi antar fitur numerik dilakukan untuk mengidentifikasi kekuatan dan arah hubungan antar variabel.

![Korelasi_variabel_numerikal](img/Korelasi_variabel_numerikal.png)  
Berdasarkan output heatmap korelasi, terdapat korelasi positif antara usia dan charges (0.30), yang menunjukkan bahwa biaya asuransi cenderung meningkat seiring bertambahnya usia. BMI (0.20) dan jumlah anak (0.07) juga memiliki korelasi positif namun sangat lemah terhadap charges, menandakan bahwa pengaruh keduanya terhadap biaya asuransi relatif kecil. Selain itu, korelasi antar variabel independen juga rendah, sehingga tidak ditemukan indikasi multikolinearitas yang signifikan dalam data.

## Data Preparation

Tahapan data preparation sangat penting untuk memastikan bahwa data yang digunakan dalam proses pemodelan bersih, relevan, dan berada dalam format yang dapat diproses oleh algoritma machine learning. Pada proyek ini, langkah-langkah data preparation dilakukan secara sistematis dan berurutan sebagai berikut:

### 1. Menangani Outliers

Untuk memastikan kualitas data yang baik sebelum pelatihan model, langkah pertama adalah menangani outlier menggunakan metode **IQR (Interquartile Range)**. Langkah ini dilakukan dengan menghitung kuartil pertama (Q1) dan kuartil ketiga (Q3), kemudian menentukan batas bawah dan batas atas menggunakan rumus:

* **Lower Bound** = Q1 - 1.5 × IQR
* **Upper Bound** = Q3 + 1.5 × IQR

![Outliner Sesudah](img/outliner_sesudah.png)  
Data yang berada di luar batas ini dianggap sebagai outlier dan dihapus. Setelah penghapusan, visualisasi dengan boxplot dilakukan untuk memastikan bahwa data telah bersih dari pencilan. Hasilnya, jumlah sampel data berkurang dari 1338 menjadi 1191 baris, yang membantu menciptakan distribusi data yang lebih stabil dan representatif untuk pelatihan model.

### 2. Encoding Fitur Kategorikal

Untuk memungkinkan algoritma machine learning memproses data kategorikal, dilakukan transformasi sebagai berikut:

* Kolom sex dan smoker diencoding secara biner:

  * sex: female → 0, male → 1
  * smoker: no → 0, yes → 1

* Kolom region memiliki lebih dari dua kategori sehingga diterapkan one-hot encoding tanpa menghapus dummy pertama (drop\_first=False) untuk mempertahankan interpretabilitas model.

### 3. Pembagian Dataset

Dataset kemudian dibagi menjadi data latih dan data uji dengan rasio 80:20 menggunakan fungsi train_test_split dari scikit-learn. Langkah ini bertujuan untuk mengevaluasi kinerja model secara adil dan menghindari overfitting.

Hasil pembagian:

* Jumlah data pada data latih: 952
* Jumlah data pada data uji : 239

### 4. Standardisasi Fitur Numerik

Kolom numerik seperti age, bmi, dan children distandarisasi menggunakan StandardScaler agar memiliki skala yang seragam. Langkah ini sangat penting terutama untuk algoritma seperti K-Nearest Neighbor atau Gradient Boosting yang sensitif terhadap skala fitur.

## Modeling

Pada tahap ini, dilakukan pembangunan model regresi untuk memprediksi biaya asuransi (charges) berdasarkan fitur-fitur demografis dan gaya hidup. Tiga algoritma machine learning diterapkan dan dibandingkan performanya, yaitu:

* K-Nearest Neighbors (KNN) Regressor
* Random Forest Regressor
* Gradient Boosting Regressor

Pemilihan ketiga model ini bertujuan untuk membandingkan performa algoritma instance-based, ensemble berbasis bagging, dan ensemble berbasis boosting.

**Tahap Modeling**

* Menyiapkan DataFrame untuk Analisis Masing-Masing Model

```python
models = pd.DataFrame(index=['train_mse', 'test_mse'],
                      columns=['KNN', 'RandomForest', 'GradientBoosting'])
```

Langkah pertama adalah membuat sebuah DataFrame bernama models yang akan digunakan untuk mencatat nilai Mean Squared Error (MSE) baik untuk data pelatihan maupun data pengujian dari tiga model regresi: KNN, Random Forest, dan Gradient Boosting.

### 1. K-Nearest Neighbors (KNN)

KNN merupakan algoritma non-parametrik yang tidak membentuk model eksplisit. Proses prediksi dilakukan dengan mencari k tetangga terdekat (dalam kasus ini k=5) dari titik data baru berdasarkan jarak Euclidean, lalu menghitung rata-rata nilai charges dari tetangga tersebut.

Model ini bekerja baik untuk data dengan struktur lokal dan non-linear, namun sangat sensitif terhadap skala fitur dan noise. Oleh karena itu, fitur numerik distandarisasi sebelum pelatihan.

```python
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
```

### 2. Random Forest Regressor

Random Forest merupakan teknik ensemble berbasis bagging yang membangun banyak pohon keputusan (decision trees) pada subset acak dari data dan fitur. Setiap pohon dilatih secara independen, dan prediksi akhir diperoleh dari rata-rata seluruh pohon. Dengan pendekatan ini, Random Forest cenderung mengurangi overfitting dan mampu menangani hubungan non-linear serta fitur campuran.

Pada proyek ini, digunakan 100 pohon (n\_estimators=100) dengan kedalaman maksimum 10 (max\_depth=10).

```python
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
```

### 3. Gradient Boosting Regressor

Gradient Boosting adalah algoritma boosting yang membangun model secara bertahap (sequential), bukan paralel seperti Random Forest. Setiap pohon baru dilatih untuk memperbaiki kesalahan (residual) dari model sebelumnya. Proses optimasi dilakukan dengan menurunkan nilai fungsi loss (Mean Squared Error) menggunakan pendekatan gradient descent.

Model ini sangat baik dalam menangkap pola kompleks, namun sensitif terhadap overfitting jika jumlah estimator terlalu banyak. Oleh karena itu, digunakan parameter moderat yaitu 100 estimator (n\_estimators=100), learning\_rate=0.1, dan max\_depth=3.

```python
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb.fit(X_train, y_train)
```

**Tahapan dan Parameter yang Digunakan**

| **Model**               | **Tahapan**                                                                                                                                     | **Parameter yang Digunakan**                                                          | **Train MSE**                                 |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | --------------------------------------------- |
| **K-Nearest Neighbors** | - Inisialisasi model dengan `KNeighborsRegressor`  <br> - Latih dengan `X_train`, `y_train` <br> - Hitung dan simpan MSE pada `models`          | `n_neighbors=5`                                                                       | `models.loc['train_mse', 'KNN']`              |
| **Random Forest**       | - Inisialisasi model dengan `RandomForestRegressor` <br> - Latih dengan data pelatihan <br> - Simpan hasil MSE ke dalam `models`                | `n_estimators=100` <br> `max_depth=10` <br> `random_state=42`                         | `models.loc['train_mse', 'RandomForest']`     |
| **Gradient Boosting**   | - Inisialisasi model dengan `GradientBoostingRegressor` <br> - Latih dengan `X_train`, `y_train` <br> - Hitung dan simpan MSE ke dalam `models` | `n_estimators=100` <br> `learning_rate=0.1` <br> `max_depth=3` <br> `random_state=42` | `models.loc['train_mse', 'GradientBoosting']` |

**Kelebihan dan Kekurangan Setiap Algoritma**

Berikut adalah tabel Kelebihan dan Kekurangan dari ketiga algoritma regresi:

| **Model**               | **Kelebihan**                                                                                                                                       | **Kekurangan**                                                                                                                                         |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **K-Nearest Neighbors** | - Sederhana dan mudah diimplementasikan <br> - Tidak memerlukan asumsi distribusi data <br> - Cocok untuk data non-linear                           | - Sensitif terhadap skala dan outlier <br> - Kinerja menurun pada dataset besar karena komputasi prediksi lambat <br> - Tidak memiliki model eksplisit |
| **Random Forest**       | - Dapat menangani data numerik dan kategorikal <br> - Mengurangi overfitting melalui agregasi banyak pohon <br> - Robust terhadap outlier dan noise | - Lebih lambat dibanding model sederhana <br> - Interpretabilitas terbatas <br> - Ukuran model bisa besar                                              |
| **Gradient Boosting**   | - Performa prediktif baik walau tanpa tuning <br> - Menangani hubungan kompleks dan non-linear <br> - Bisa langsung digunakan dengan default param  | - Proses pelatihan relatif lambat <br> - Lebih kompleks dibanding Random Forest <br> - Bisa overfitting jika jumlah estimator terlalu besar            |

## Evaluation

**Mean Squared Error (MSE)** adalah metrik yang digunakan untuk mengevaluasi seberapa baik model dalam memprediksi data. MSE mengukur seberapa besar kesalahan antara nilai yang diprediksi oleh model dan nilai yang sebenarnya dengan menghitung rata-rata dari selisih kuadrat antara keduanya. Semakin kecil nilai MSE, semakin baik model dalam memprediksi data yang sebenarnya, karena model berhasil meminimalkan kesalahan prediksi. Rumus MSE adalah:

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

Keterangan:

* \$y\_i\$ adalah nilai yang sebenarnya,
* \$\hat{y}\_i\$ adalah nilai yang diprediksi oleh model,
* \$n\$ adalah jumlah data yang digunakan.


**Berikut adalah hasil evaluasi model dalam bentuk tabel:**

| **Model**             | **Train MSE** | **Test MSE**  |
| --------------------- | ------------- | ------------- |
| **KNN**               | 19,099,397.13 | 25,944,624.82 |
| **Random Forest**     | 4,524,854.71  | 19,449,671.35 |
| **Gradient Boosting** | 12,817,935.84 | 18,014,631.47 |

* Model KNN menghasilkan nilai MSE sebesar **19,099,397.13** pada data latih dan **25,944,624.82** pada data uji, menunjukkan bahwa model ini mengalami overfitting ringan dan kurang akurat dalam memprediksi data baru.
* Model Random Forest menunjukkan performa terbaik di antara ketiga model dengan nilai MSE yang paling rendah pada data latih yaitu **4,524,854.71**, dan **19,449,671.35** pada data uji.
* Model Gradient Boosting memberikan performa yang cukup baik dengan MSE sebesar **12,817,935.84** pada data latih dan **18,014,631.47** pada data uji, lebih stabil dibanding KNN namun tidak sebaik Random Forest.

![Perbandingan_MSE_Model](img/Perbandingan_MSE_Model.png) 

**Model Terbaik** 

Berdasarkan hasil evaluasi, model Gradient Boosting dipilih sebagai model terbaik karena memiliki MSE terendah pada data uji dibandingkan KNN dan performanya lebih stabil. Berikut adalah grafik untuk nilai aktual vs nilai prediksi menggunakan model Gradient Boosting.

![Visualisasi_Gradient_Boosting](img/Visualisasi_Gradient_Boosting.png)

Model Gradient Boosting menunjukkan akurasi yang baik, terutama pada *charges* di bawah 20.000, dengan prediksi mendekati garis referensi. Meski sedikit meleset pada nilai *charges* tinggi, model ini tetap memberikan keseimbangan bias dan variansi yang baik, serta MSE terendah.

**Feature Importance dari Gradient Boosting** 

Berdasarkan analisis feature importance, fitur yang paling memengaruhi biaya asuransi adalah smoker, age, dan BMI. Individu yang merokok, berusia lebih tua, dan memiliki BMI tinggi cenderung membayar biaya asuransi lebih mahal karena memiliki risiko kesehatan yang lebih besar.

![Feature_Importance](img/Feature_Importance.png)

**Evaluasi Terhadap Business Understanding**

🧩 Menjawab Problem Statement:
Model yang dibangun berhasil menjawab problem statement dengan memprediksi biaya asuransi kesehatan berdasarkan fitur-fitur seperti usia, jenis kelamin, BMI, kebiasaan merokok, jumlah anak, dan wilayah tempat tinggal. Selain itu, visualisasi feature importance dari model Gradient Boosting menunjukkan bahwa fitur seperti kebiasaan merokok dan BMI memang memiliki pengaruh signifikan terhadap besarnya biaya asuransi.

🎯 Mencapai Goals:
Model prediksi regresi yang dikembangkan—melalui K-Nearest Neighbor, Random Forest, dan Gradient Boosting berhasil mencapai tujuan utama, yaitu memberikan prediksi biaya asuransi yang cukup akurat. Model Gradient Boosting dipilih sebagai model terbaik karena memiliki nilai Mean Squared Error (MSE) terendah dan visualisasi prediksi yang paling mendekati nilai aktual. Fitur-fitur yang paling berpengaruh terhadap charges juga berhasil diidentifikasi.

🚀 Dampak dari Solution Statement:
Pendekatan menggunakan beberapa algoritma regresi memungkinkan perbandingan performa antar model, sehingga membantu dalam memilih model terbaik untuk prediksi. Penggunaan metrik MSE sebagai evaluasi memberikan gambaran yang objektif terhadap tingkat kesalahan prediksi. Secara keseluruhan, solusi yang diterapkan memberikan hasil yang efektif dalam menjawab permasalahan dan mencapai tujuan proyek.


## Kesimpulan

Proyek ini berhasil membangun model prediksi biaya asuransi kesehatan menggunakan algoritma regresi. Gradient Boosting dipilih sebagai model terbaik karena memberikan hasil prediksi paling akurat. Fitur smoker, age, dan BMI terbukti paling berpengaruh terhadap biaya asuransi. Model ini dapat digunakan untuk membantu perusahaan asuransi dalam menetapkan premi secara lebih objektif dan efisien.
