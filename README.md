# Predict-Telco-Customer-Churn

- **Tools** : Jupyter Notebook | [View](www.Hehehehehehe) <br>
- **Programming Language** : Python <br>
- **Library** : Pandas, Numpy, Sklearn, Tensorflow, Keras <br>
- **Visualization** : Matplotlib, Seaborn <br>
- **Source Dataset** : Binar Academy <br>
- **Presentation Deck** : [View](www.hahahahah) <br> <br>

**Table Of Contents**
- [Business Understanding](https://github.com/EndowBonapen/Telco-Customer-Churn/blob/main/README.md#-business-understanding)
- [Workflow](https://github.com/EndowBonapen/Telco-Customer-Churn/blob/main/README.md#-workflow)
- [Data Understanding](https://github.com/EndowBonapen/Telco-Customer-Churn/blob/main/README.md#-data-understanding)
- [Exploratory Data Analysis (EDA)](https://github.com/EndowBonapen/Telco-Customer-Churn/blob/main/README.md#-exploratory-data-analysis-(EDA))
- [Data Preprocessing](https://github.com/EndowBonapen/Telco-Customer-Churn/blob/main/README.md#-data-preprocessing)
- [Modeling and Evaluation](https://github.com/EndowBonapen/Telco-Customer-Churn/blob/main/README.md#-modeling-and-evaluation)
- [Business Recomendation](https://github.com/EndowBonapen/Telco-Customer-Churn/blob/main/README.md#-business-recomendation)
<br>

## 📂 Business Understanding
### Problem Statement
**Customer churn** menggambarkan keadaan pelanggan suatu bisnis atau layanan yang menghentikan hubungan 
mereka dengan perusahaan tersebut akibat ketidakpuasan pelanggan, harga yang tidak kompetitif, pengalaman 
pengguna yang buruk, atau penawaran yang lebih baik dari pesaing. Peningkatan customer churn dapat 
**berdampak negatif** pada pendapatan dan reputasi perusahaan karena biaya akuisisi pelanggan baru sering kali 
lebih tinggi daripada mempertahankan pelanggan yang ada.
<br>
### Goals
1. Membangun model machine learning dan deep learning yang dapat digunakan untuk memprediksi **Customer Churn**.
2. Membandingkan beberapa algoritma guna memperoleh akurasi terbaik dalam melakukan prediksi terhadap **Customer Churn**.
3. Mengidentifikasi variabel yang paling efektif dalam menentukan **Customer Churn**. 

### Solution Statements
Untuk mencapai tujuan yang telah ditetapkan, peneliti mengembangkan model prediktif menggunakan 6 (enam) algoritma yang berbeda. Setiap model akan dievaluasi secara komprehensif untuk menentukan model yang paling optimal dalam memprediksi **customer churn**. Berikut adalah algoritma yang akan digunakan dalam pembangunan model prediksi:

1. **Random Forest** <br>
Random Forest adalah sebuah algoritma machine learning yang bekerja dengan menggabungkan output dari beberapa decision tree untuk menghasilkan 1 (satu) model prediktif yang optimal. Random Forest terkenal dengan kemudahan penggunaan dan fleksibilitasnya, sehingga banyak diminati.  Algoritma ini dapat menangani permasalahan klasifikasi maupun regresi.
2. **XGB Classifier**<br>
XGB Classifier adalah toolkit distributed gradient boosting yang telah disesuaikan untuk pelatihan yang efisien dan scalable dari model machine learning. Algoritma ini menggunakan decision trees sebagai base learners dan menerapkan teknik regularisasi untuk meningkatkan generalisasi model. Dikenal karena efisiensi komputasinya, analisis pentingnya variabel, dan penanganan nilai-nilai yang hilang, XGB Classifier banyak digunakan untuk tugas-tugas seperti regresi, klasifikasi, dan ranking.
3. **LightGBM Classifier**<br>
LightGBM Classifier adalah metode dari kerangka Gradient Boosting yang cepat, terdistribusi dan berkinerja tinggi berdasarkan algoritma pohon keputusan yang dapat
digunakan untuk peringkat, klasifikasi, regresi dan banyak tugas pembelajaran mesin lainnya.
4. **Decision Tree**<br>
salah satu cara data processing dalam memprediksi masa depan dengan cara membangun klasifikasi atau regresi model dalam bentuk struktur pohon. Hal tersebut dilakukan dengan cara memecah terus ke dalam himpunan bagian yang lebih kecil lalu pada saat itu juga sebuah pohon keputusan secara bertahap dikembangkan. Hasil akhir dari proses tersebut adalah pohon dengan node keputusan dan node daun.
5. **Support Vector Machine (SVM)**<br>
Support Vector Machine (SVM) adalah satu metode dalam supervised learning yang biasanya digunakan untuk klasifikasi (seperti Support Vector Classification) dan regresi (Support Vector Regression). Support Vector Machine (SVM) digunakan untuk mencari hyperplane terbaik dengan memaksimalkan jarak antar kelas. Hyperplane adalah sebuah fungsi yang dapat digunakan untuk pemisah antar kelas. Dalam 2-D fungsi yang digunakan untuk klasifikasi antar kelas disebut sebagai line whereas, fungsi yang digunakan untuk klasifikasi antas kelas dalam 3-D disebut plane similarly, sedangan fungsi yang digunakan untuk klasifikasi di dalam ruang kelas dimensi yang lebih tinggi di sebut hyperplane.
6. **Deep Neural Network (DNN)**<br>
Deep Neural Network (DNN) adalah model komputasi yang terinspirasi dari struktur dan fungsi jaringan saraf biologis dalam otak manusia [12]. Tujuan utama Neural Network adalah untuk memproses informasi dan melakukan tugas-tugas seperti klasifikasi, regresi, pengenalan pola, dan lainnya, dengan cara yang mirip dengan cara otak manusia memproses informasi.

### Objective
1. **Data Collection** : Mengumpulkan data historis, seperti demografi pelanggan, riwayat transaksi, dan interaksi.
2. **Feature Engineering** : Mengidentifikasi dan mengekstrak fitur-fitur yang dapat memengaruhi pelanggan,  seperti pola penggunaan dan interaksi dukungan pelanggan.
3. **Model Development** : Membangun algoritma machine learning untuk prediksi.
4. **Model Evaluation** : Mengevaluasi kinerja model prediktif menggunakan metrik yang tepat diikuti dengan validasi model. <br>

## 📂 Workflow
<div align="center">
  <img src="https://drive.google.com/uc?id=10SLhtfi_1n_uqQa-dfIgsgFXTRl-zr6m" alt="Workflow">
  <p>Gambar 1. Workflow Pembuatan Model.</p>
</div> 

## 📂 Data Understanding
### Data Overview
Dataset memiliki total data 4250 record dengan 20 fitur yaitu (5 fitur kategorikal dan 15 fitur numerical) <br>
Tabel 1 — Deskripsi Fitur
Fitur | Deskripsi
------|----------
State | Status tempat tinggal utama Customer
Account Length | Total bulan Customer aktif berlangganan pelayanan telco provider
Area Code | Kode Area tempat tinggal Customer
International Plan | Customer Berlangganan Paket International Plan 
Voice Mail Plan | Customer Berlangganan Paket Voice Mail Plan 
Number Vmail Messages | Total pesan Voice Mail
Total Day Minutes | Total panggilan dalam satu hari per menit
Tota Day Calls | Total panggilan dalam satu hari
Total Day Charge | Total charge dalam panggilan perhari
Total Eve Minutes | Total panggilan di waktu sore hari per menit
Tota Eve Calls | Total panggilan dalam waktu sore hari
Total Eve Charge | Total charge dalam panggilan sore hari
Total Night Minutes | Total panggilan di waktu malam hari per menit
Tota Night Calls | Total panggilan dalam waktu malam hari
Total Night Charge | Total charge dalam panggilan malam hari
Total Intl Minutes | Total International Call per menit
Tota Intl Calls | Total International Call
Total Intl Charge | Total charge dalam International Call
Number Customer Service Call | Total panggilan kepada Customer Service
## 📂 Exploratory Data Analysis
Exploratory Data Analysis (EDA) adalah proses investigasi awal yang dilakukan pada dataset untuk memahami dan menganalisis karakteristik utama dalam dataset. Tujuan dari EDA adalah untuk mengidentifikasi pola, hubungan, anomali, dan informasi penting lainnya dalam dataset tanpa membuat asumsi atau hipotesis terlebih dahulu. Metode yang umum digunakan dalam EDA meliputi visualisasi data, statistik deskriptif, dan teknik analisis lainnya untuk mendapatkan pemahaman yang mendalam tentang data sebelum melakukan analisis lebih lanjut atau membangun model prediktif. <br>
### 1. Churning Ratio
Rata-rata persentase customer churn pada perusahaan telco berkisar antara **21-22%** dan besar customer churn pada perusahaan ini berada pada angka **14%** sehingga masih tergolong **normal**, dapat dilihat pada gambar dibawah ini:
<div align="center">
  <img src="https://drive.google.com/uc?id=1NTiti4vBLnJSpkC77mQqgCs-ngyjjjYJ" alt="Workflow">
  <p>Gambar 2. Churning Ratio.</p>
</div>


### 2. Univariative Analysis
Univariate Analysis adalah sebuah metode analisis statistik yang digunakan untuk memahami karakteristik dari satu variabel tunggal dalam suatu dataset. Tujuan utama dari analisis univariat adalah untuk merangkum dan menyajikan data, serta mendapatkan wawasan yang lebih dalam tentang distribusi, pola, dan sifat-sifat statistik dari variabel.
### Distribution of Data
<div align="center">
  <img src="https://drive.google.com/uc?id=1s1-10YibFLJ6jXw7t-NeJqKLr7wtjUOW" alt="Workflow">
  <p>Gambar 3. Distribusi Data.</p>
</div>


**Key Takeaways:**
1. Mayoritas keseluruhan data terdistribusi normal.
2. Fitur **number_vmail_massages**, **total_intl_calls** dan **number_customer_sevice_calls** memiliki distribusi positively skewed yang dapat dilihat dengan ciri yaitu pada nilai **Mean** (rata-rata) lebih besar daripada nilai **Median** dan Nilai **Median** lebih besar daripada Nilai **Modus**.

### 3. Bivariative Analysis
### 4. Multivariative Analysis



