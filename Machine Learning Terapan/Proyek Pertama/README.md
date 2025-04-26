# Laporan Proyek Machine Learning - Ikrar Bagaskara

---

## Domain Proyek

---

Dalam era digital dan informasi saat ini, pasar saham menjadi salah satu instrumen investasi yang sangat diminati oleh berbagai kalangan, mulai dari individu hingga institusi besar. Terlebih bagi para milenial atau Gen-Z yang mulai melek permasalahan ekonomi, pasar saham selalu menjadi perbincangan hangat. Pergerakan harga saham yang dinamis menawarkan peluang keuntungan yang besar, namun juga diiringi oleh risiko yang tinggi. Oleh karena itu, kemampuan untuk memprediksi harga saham secara akurat menjadi hal yang sangat berharga.

Menurut penelitian yang dilakukan oleh [[1]](http://link.springer.com/10.1007/s11227-017-2228-y) menunjukan bahwa *machine learning* dapat membantu prediksi harga saham secara efektif, dibandingkan strategi *trading tradisional* meskipun begitu prediksi harga saham menggunakan *machine learning* hanya bersifat pembantu keputusan, dan tidak bisa dijadikan alat tunggal untuk menentuka keputusan akhir. Dikarenakan banyakanya faktor eksternal yang dapat mempengaruhi hasil akhir dalam pengambilan keputusan, apalagi meyangkut masalah ekonomi seperti saham ini.

Pada studi lain yang dilakukan oleh [[2]](https://arxiv.org/pdf/2003.01859) menunjukan progress dari *machine learning* pada bidang financial atau ekonomi semakin membaik dengan adanya teknologi atau algoritma, yang berarti *machine learning* dapat semakin diandalkan terutama pada bidang ekonomi.

#### Mengapa masalah ini penting?

* **Pengambilan keputusan:** Prediksi harga saham dapat membantu investor untuk mengambil keputusan beli, jual, atau tahan terhadap suatu aset.
* **Penghematan biaya dan waktu**: Prediksi harga saham secara manual memakan banyak sumber daya, apalagi jika menganalisis banyak saham. Dengan prediksi otomatis, proses ini lebih efisien, mengurangi beban, dan memungkinkan fokus pada tugas lainnya.
* **Manajemen risiko:** Dengan memiliki gambaran tentang kemungkinan pergerakan harga, pelaku pasar dapat melakukan mitigasi risiko lebih awal.
* **Penerapan teknologi:** Proyek ini menunjukkan bagaimana teknologi terkini seperti LSTM dan GRU dapat dimanfaatkan dalam dunia nyata, khususnya dalam domain keuangan.

## Business Understanding

---

### Problem Statements

Berdasarkan latar belakang di atas, beberapa permasalahan yang dapat dirumuskan adalah sebagai berikut:

* Bagaimana cara memprediksi harga saham secara akurat untuk membantu pengambilan keputusan investasi yang lebih baik?
* Dapatkah model prediksi harga saham memberikan hasil yang stabil dan dapat diandalkan meskipun hanya dengan data historis terbatas?
* Bagaimana memilih arsitektur model yang paling tepat untuk menangani data time series dengan karakteristik fluktuatif seperti harga saham?

### Goals

Untuk mencapai tujuan di atas, beberapa solusi yang diusulkan adalah:

* Membangun model prediksi harga saham yang dapat digunakan untuk membantu investor dalam mengambil keputusan investasi yang lebih tepat.
* Menguji seberapa baik model dapat memberikan prediksi yang stabil meskipun data yang digunakan terbatas.
* Menentukan arsitektur model terbaik yang sesuai untuk memprediksi data saham harian.

### Solution statements

- Membangun model prediksi harga saham menggunakan arsitektur **LSTM (Long Short-Term Memory)** dan **GRU (Gated Recurrent Unit)**, yang keduanya dirancang untuk menangani data *time series* dan dapat memahami pola historis pergerakan harga saham.

* Menilai kinerja model dengan menggunakan metrik evaluasi yang sesuai, seperti  **MAE (Mean Absolute Error)**, **MSE (Mean Squared Error)**, **RMSE (Root Mean Squared Error)**, dan **R²**, untuk memastikan model dapat memberikan prediksi yang akurat dan dapat diandalkan.
* Melakukan evaluasi perbandingan antar arsitektur model untuk menentukan mana yang paling efektif dalam konteks data saham.

## Data Understanding

---

Dalam proyek ini, digunakan dataset [Microsoft Stock Data ](https://www.kaggle.com/datasets/zongaobian/microsoft-stock-data-and-key-affiliated-companies?select=MSFT_daily_data.csv)yang diambil langsung dari Kaggle. Dataset ini memuat data historis saham perusahaan teknologi raksasa, yaitu **Microsoft**, dengan rentang waktu mulai dari **13 Maret 1986** hingga **30 Oktober 2024**, dan interval data harian. Data yang digunakan mencakup beberapa fitur pasar saham seperti *Open, High*, *Low,* *Close*, *Volume*, dan *Adjusted Close*. Namun, fokus utama dalam proyek ini adalah pada fitur **harga penutupan harian (Close)**  yang dianggap merepresentasikan nilai akhir saham dalam satu hari perdagangan.

Tujuan dari penggunaan dataset ini adalah untuk membangun model yang mampu memperkirakan harga saham Microsoft di masa depan berdasarkan pola historis dari harga tutup tersebut.

### Deskripsi Dataset

* Dataset ini terdiri dari 9737 baris data dengan 6 fitur numerikal dan 1 fitur object.
* Dataset tidak memliki fitur kategorikal sehingga tidak perlu dilakukan *encoding*.

#### Kondisi Dataset

* Memiliki outlier yang wajar karena adanya fluktuasi harga saham pada waktu tertentu, sehingga tidak dihapus.
* Tidak memiliki tidak memiliki *missing value,* sehingga semua kolom memiliki isi yang lengkap.

#### Fitur Dataset

| Nama Fitur | Deskripsi                                                                                                            |
| ---------- | -------------------------------------------------------------------------------------------------------------------- |
| Date       | Tanggal spesifik untuk data saham.                                                                                   |
| Open       | Harga saham Microsoft saat pasar dibuka pada hari tersebut.                                                          |
| High       | Harga tertinggi saham yang dicapai selama satu hari perdagangan.                                                     |
| Low        | Harga terendah saham yang dicapai selama satu hari perdagangan.                                                      |
| Close      | Harga saham saat pasar ditutup pada hari tersebut. Fitur ini menjadi fokus utama dalam peramalan harga saham.        |
| Adj Close  | Harga penutupan yang telah disesuaikan untuk dividen, split saham, dan penyesuaian korporasi lainnya.                |
| Volume     | Jumlah total saham yang diperdagangkan pada hari tersebut. Mengindikasikan aktivitas pasar dan minat terhadap saham. |

### Analisis Univariate

1. Distribusi Data
   ![univariate-1](https://github.com/user-attachments/assets/d94a9e3e-2053-442d-9ef4-0efeb6e242b7)
   Berdasarkan distribusi diatas, sebagain besar harga sahama berada di level atau harga rendah, meskipun begitu harga saham juga meningkatkan secara signifikan dalam jangka waktu panjang.
2. Tren Harian Harga Penutupan
   ![univariate-2](https://github.com/user-attachments/assets/6543caa6-6d2a-470d-bea1-dd97aad04b9c)
   Harga saham mengalami pertumbuhan jangkan panjang yang kuat bahkan pada beberapa periode lonjakan naik secara tajam namun dengan beberapa periode yang menunjukan harga saham masih stagnan .

### Analisis Multivariate

1. Analisa Korelasi Antar Fitur

   ![multivariate-1](https://github.com/user-attachments/assets/e13c8572-101f-4bd4-bcd1-d6afbf445c26)

   Fitur `Open`, `High`, `Low`, `Close`, dan `Adj Close` memiliki korelasi sempurna (1.00) yang bisa menunjukan bahwa kelima fitur tersebut **overlapped**, dan jika digunakan bersamaan dapat menyebabkan redudansi fitur, oleh karena       itu pada model predictive ini hanya fitur `Close` yang akan digunakan sebagai target.
2. Scatter Plot: Volume vs Close

   ![multivariate-2](https://github.com/user-attachments/assets/6405166e-82a0-42f2-92cf-011c66e4bbbc)

   Tidak tampak pola linier yang kuat antara harga penutupan (Close) dengan Volume. Yang mungkin volume hanya mencerminkan aktivitas pasar saja tetapi tidak mempengaruhi arah harga baik turun ataupun naik.

## Data Preparation

---

Pada tahap ini, dilakukan beberapa langkah untuk mempersiapkan data agar sesuai dengan kebutuhan model time series berbasis deep learning, khususnya LSTM dan GRU

### Pemilihan Fitur

Walaupun awalnya dataset memiliki berbagai fitur, namun dalam proyek ini hanya digunakan kolom `Date` dan `Close`, karena **harga penutupan (Close)** dianggap sebagai representasi paling relevan dalam memprediksi pergerakan harga saham di masa depan.

### Penetapan Index Waktu

```python
df_model = df[['Date', 'Close']]
df_model = df_model.set_index('Date')
```

Data time series harus memiliki urutan waktu yang eksplisit, sehingga kolom `Date` dijadikan index untuk menjaga urutan data.

### Normalisasi Data

```python
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df_model)
```

Normalisasi dengan **MinMaxScaler** digunakan agar seluruh nilai berada dalam rentang 0 hingga 1, agar dapat mempercepat proses pelatihan dan membantu model memahami pola dari data

### Pembuatan Sequence Data

```python
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

sequence_length = 60
X, y = create_sequences(df_scaled, sequence_length)
```

Karena model akan memprediksi harga berdasarkan *n* hari sebelumnya, data diubah menjadi *sequence* atau urutan dengan panjang 60 hari. Ini artinya, model belajar memprediksi harga pada hari ke-61 berdasarkan 60 hari sebelumnya.

### Data Splitting

> ```python
> train_size = int(len(X) * 0.8)
> X_train, y_train = X[:train_size], y[:train_size]
> X_test, y_test = X[train_size:], y[train_size:]
> ```

Data dibagi dengan rasio 80:20 untuk training dan testing agar model diuji pada data yang belum pernah dilihat sebelumnya. Hal ini penting untuk mengukur kemampuan generalisasi model terhadap data baru.

## Modeling

---

Pada tahap ini, dilakukan pembangunan dan pelatihan model deep learning untuk melakukan prediksi harga saham Microsoft berdasarkan data historis. Mengingat data bersifat urutan waktu *time series*, model yang digunakan adalah jenis **Recurrent Neural Network (RNN),** khususnya:

* **Long Short-Term Memory (LSTM)**
* **Gated Recurrent Unit (GRU)**

Kedua model ini dipilih karena memiliki keunggulan dalam menangkap pola jangka panjang pada data sekuensial seperti harga saham.

### **Long Short-Term Memory (LSTM)**

LSTM adalah jenis **Recurrent Neural Network (RNN)** yang dirancang untuk mengatasi masalah *vanishing gradient* yang sering terjadi pada RNN standar. Model ini memiliki arsitektur khusus berupa **gate (gerbang)** seperti input gate, forget gate, dan output gate, yang memungkinkan LSTM untuk menyimpan atau melupakan informasi tertentu dari waktu ke waktu [[3]](http://arxiv.org/abs/1810.10161).

**Detail Arsitektur dan Parameter:**

* 3 lapisan LSTM dengan 64 dan 32 unit, bertujuan menangkap pola temporal dari data.
* 2 lapisan `Dense` untuk transformasi akhir ke output (harga).
* Fungsi aktivasi default ( *tanh* ) pada LSTM.
* **Loss function** : `mean_squared_error`
* **Optimizer** : `Adam`

| Kelebihan                                                                 | Kekurangan                                                                                                                             |
| ------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Presisi tinggi untuk data kompleks, terutama dalam konteks prediksi saham | Butuh data lebih banyak untuk training efektif. Oleh karena itu pada kasus ini hasil**LSTM** lebih burk daripada **GRU**  |
| Cocok dan stabil untuk data jangka panjang                                | Training lebih lambat & berat karena banyak parameter                                                                                  |

### Gated Recurrent Unit (GRU)

**GRU** adalah varian dari RNN yang mirip dengan LSTM, tetapi memiliki arsitektur yang lebih sederhana karena hanya menggunakan dua jenis gate: **update gate** dan **reset gate** [[4]](https://linkinghub.elsevier.com/retrieve/pii/S0960077921002149)**.**

* **Update gate** bertugas untuk menentukan seberapa banyak informasi dari waktu sebelumnya yang akan dibawa ke waktu sekarang.
* **Reset gate** mengatur seberapa banyak informasi dari waktu sebelumnya yang akan dilupakan.

**Detail Arsitektur dan Parameter:**

* 1 lapisan GRU dengan 50 unit sebagai pemroses utama *sequence* data.
* 1 lapisan Dense sebagai output layer.
* **Loss function** : `mean_squared_error`
* **Optimizer** : `Adam`

| Kelebihan                                           | Kekurangan                                                                                                          |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Cocok untuk data sederhana atau data jangka pendek | Kurang kuat untuk pola sangat panjang                                                                               |
| Efisien dengan dataset kecil                        | GRU kurang fleksibel dalam mengatur informasi dibanding LSTM. Karena jumlah gerbang yang lebih sedikit daripada GRU |

### Training Parameter

* **Epochs** : 20
* **Batch Size** : 32
* **Validation Data** : Menggunakan 20% dari dataset
* **Callbacks** :
  * `EarlyStopping` untuk menghentikan pelatihan jika validasi tidak membaik.
  * `ReduceLROnPlateau` untuk mengurangi learning rate jika model stagnan.

### Model Akhir

GRU dipilih sebagai model akhir dengan nilai evaluasi lebih baik daripada LSTM, bahkan saat pelatihan dan evaluasi model GRU juga jauh lebih cepat mengungguli LSTM. Pada skenario ini, *sequence window* yang dipilih juga hanya 60 hari terakhir, sehingga model hanya mempelajari data dalam jangka pendek. Oleh karena itu GRU lebih baik dalam kasus ini. Terlampir pada gambar dibawah, prediksi dengan menggunakan model GRU (warna hijau) menghasilkan prediksi yang hampir sesuai dengan nilai asli dari harga saham daripada menggunakan model LSTM (warna merah).

![evaluasi_prediksi](https://github.com/user-attachments/assets/1d6a1ff1-a66e-4c94-9024-6eddb2e8fa2c)

## Evaluation

---

### Matrik Evaluasi

**Mean Absolute Error (MAE) :** Rata-rata dari selisih absolut antara nilai aktual dan nilai prediksi.

**Formula**:

$$
MAE = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \right|
$$

**Penjelasan Formula:**

* $y_i$ = nilai aktual ke-$i$
* $\hat{y}_i$ = nilai prediksi ke-$i$
* $|y_i - \hat{y}_i|$ = selisih absolut
* $\sum$ = penjumlahan seluruh data
* $\frac{1}{n}$ = rata-rata

**Catatan:** Semakin kecil MAE, semakin akurat model.

**Mean Squared Error (MSE) :** Rata-rata dari kuadrat selisih antara nilai aktual dan prediksi.

**Formula:**

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} \left( Y_i - \hat{Y}_i \right)^2
$$

**Penjelasan Formula:**

* $y_i$ = nilai aktual ke-$i$
* $\hat{y}_i$ = nilai prediksi ke-$i$
* `(y_i - \hat{y}_i)^2` = selisih dikuadratkan
* $\sum$ = jumlahkan semua squared error
* $\frac{1}{n}$ = rata-rata dari semua squared error

**Catatan:** Semakin kecil MSE, semakin baik performa model.

**Root Mean Squared Error (RMSE) :** Akar dari MSE. Mengembalikan nilai kesalahan dalam satuan yang sama dengan target.

**Formula:**

$$
RMSE = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2 }
$$

**Penjelasan Formula:**

* $y_i$ = nilai aktual ke-$i$
* $\hat{y}_i$ = nilai prediksi ke-$i$
* $(y_i - \hat{y}_i)^2$ = selisih dikuadratkan
* $\sum$ = jumlahkan semua squared error
* $\frac{1}{n}$ = rata-rata squared error
* $\sqrt{}$ = akar kuadrat untuk mengembalikan ke satuan asli

**Catatan:** Nilai kecil menunjukkan prediksi yang dekat dengan nilai aktual.

**R-Squared Error (R²) :** Mengukur proporsi variasi pada data yang bisa dijelaskan oleh model.

**Formula:**

$$
R^2 = 1 - \frac{ \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 }{ \sum_{i=1}^{n} (y_i - \bar{y})^2 }
$$

**Penjelasan Formula:**

* $y_i$ = nilai aktual ke-$i$
* $\hat{y}_i$ = nilai prediksi ke-$i$
* $\bar{y}$ = rata-rata dari semua nilai aktual
* $\sum (y_i - \hat{y}_i)^2$ = jumlah squared error dari model (residual sum of squares)
* $\sum (y_i - \bar{y})^2$ = total variasi data aktual (total sum of squares)
* $R^2$ = proporsi variasi data yang berhasil dijelaskan oleh model

**Catatan:**

* R² = 1 → prediksi sempurna
* R² = 0 → model tidak lebih baik dari rata-rata
* R² < 0 → model lebih buruk dari rata-rata

### Hasil Evaluasi

| Model          | MAE    | MSE      | RMSE    | R²    |
| -------------- | ------ | -------- | ------- | ------ |
| **LSTM** | 6.9762 | 110.5113 | 10.5124 | 0.9909 |
| **GRU**  | 3.3612 | 23.3751  | 4.8348  | 0.9981 |

Catatan: nilai akhir mungkin akan berbeda ketika dijalankan ulang

### Dampak Model Terhadap *Business Understanding*

#### Problem Statements

* Bagaimana cara memprediksi harga saham secara akurat untuk membantu pengambilan keputusan investasi yang lebih baik?
  * Dengan membangun model LSTM dan GRU berbasis *time series* , menggunakan data historis 60 hari sebagai input dan pendekatan *sliding window*, lalu mengevaluasi akurasi dengan metrik R², MAE, MSE, dan RMSE dan melakukan perbandingan untuk menentukan model akhir yang dipilih.
* Dapatkah model prediksi harga saham memberikan hasil yang stabil dan dapat diandalkan meskipun hanya dengan data historis terbatas?
  * Kedua model yang diajukan, yaitu **GRU** dan **LSTM** mampun memprediksi harga saham secara akurat dan stabil dengan data historis yang terbatas, bahkan dengan hanya menggunakan data historis 60 hari, kedua model mampu mendapatkan evaluasi akhir untuk **R²** di atas 0.99 atau mendekati sempurna.
* Bagaimana memilih arsitektur model yang paling tepat untuk menangani data time series dengan karakteristik fluktuatif seperti harga saham?
  * Berdasarkan hasil evaluasi menggunakan  **MAE, MSE, RMSE,** dan **R²** , kedua model memiliki performa yang baik, namun **GRU** sedikit lebih unggul dari **LSTM** dalam efisiensi pelatihan dan nilai error yang lebih rendah. Oleh karena itu, **GRU** dapat direkomendasikan untuk menangani data time series dalam waktu historis pendek seperti kasus submission ini adalah 60 hari.

#### Goals

* Membangun model prediksi harga saham yang dapat digunakan untuk membantu investor dalam mengambil keputusan investasi yang lebih tepat.

  * Tercapai, karena model mampu menghasilkan prediksi yang sangat mendekati nilai aktual dan dapat dimanfaatkan sebagai bahan pertimbangan investasi.
* Menguji seberapa baik model dapat memberikan prediksi yang stabil meskipun data yang digunakan terbatas.

  * Tercapai, model mampu mempertahankan kestabilan dan performa meskipun hanya menggunakan data 60 hari terakhir, sesuai skenario *windowed sequence* yang dipilih.
* Menentukan arsitektur model terbaik yang sesuai untuk memprediksi data saham harian.

  * Tercapai, melalui evaluasi perbadingan antara LSTM dan GRU menggunakan metrik seperti MAE, MSE, MSRE, dan R² serta komparasi dalam bentuk visualisasi (line chart). GRU dipilih sebagai model yang lebih efisien dan akurat dalam kasus ini.

### Solution statements

- Membangun model prediksi harga saham menggunakan arsitektur **LSTM (Long Short-Term Memory)** dan **GRU (Gated Recurrent Unit)**, yang keduanya dirancang untuk menangani data *time series* dan dapat memahami pola historis pergerakan harga saham.
  - Berdampak, karena kedua arsitektur tersebut, yaitu LSTM dan GRU berhasil mengenali pola historis dengan akurat, dan model-model tersebut memang cocok untuk menangani data *time series*

* Menilai kinerja model dengan menggunakan metrik evaluasi yang sesuai, seperti  **MAE (Mean Absolute Error)**, **MSE (Mean Squared Error)**, **RMSE (Root Mean Squared Error)**, dan **R²**, untuk memastikan model dapat memberikan prediksi yang akurat dan dapat diandalkan.
  * Berdampak, dengan metrik-metrik tersebut kita dapat mengetahui nilai akurasi atau performa model dan dapat menentukan apakah model yang dikembangkan dapat diandalkan untuk prediksi atau tidak.
* Melakukan evaluasi perbandingan antar arsitektur model untuk menentukan mana yang paling efektif dalam konteks data saham.
  * Berdampak, dengan melakukan evalusasi komparatif secara menyeluruh, dapat diketahui bahwa model GRU lebih akurat dan efisien untuk skenario atau konteks data saham dengan data historis terbatas.

## Kesimpulan

---

Model **GRU** menunjukkan performa lebih baik dibanding LSTM di semua metrik evaluasi. Dengan kesalahan yang lebih kecil dan akurasi yang lebih tinggi (R² mendekati 1), walaupun begitu model tetap perlu ditingkatkan karena berhubungan langsung dengan finansial yang dimana kesalahan kecil dapat berdampak besar seperti kerugian perusahaan.

## Referensi

---

[1] X. Pang, Y. Zhou, P. Wang, W. Lin, dan V. Chang, “An innovative neural network approach for stock market prediction,”  *J. Supercomput.* , vol. 76, no. 3, hlm. 2098–2118, Mar 2020, doi: 10.1007/s11227-017-2228-y.

[2] W. Jiang, “Applications of deep learning in stock market prediction: Recent progress,”  *Expert Syst. Appl.* , vol. 184, hlm. 115537, Des 2021, doi: 10.1016/j.eswa.2021.115537.

[3] R. A. Sunan, H. F. E. K., dan C. S. K. Aditya, “Klasifikasi Hoax Berita Politik Menggunakan Algoritma Long Short-Term Memory (LSTM) dengan Penambahan Fitur Embedding Global Vector (GloVe),”  *J. Edukasi Dan Penelit. Inform. JEPIN* , vol. 10, no. 2, hlm. 287, Agu 2024, doi: 10.26418/jp.v10i2.76042.

[4] K. E. ArunKumar, D. V. Kalaga, Ch. M. S. Kumar, M. Kawaji, dan T. M. Brenza, “Forecasting of COVID-19 using deep layer Recurrent Neural Networks (RNNs) with Gated Recurrent Units (GRUs) and Long Short-Term Memory (LSTM) cells,”  *Chaos Solitons Fractals* , vol. 146, hlm. 110861, Mei 2021, doi: 10.1016/j.chaos.2021.110861.
