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
- [Insight](https://github.com/EndowBonapen/Telco-Customer-Churn/blob/main/README.md#-insight)
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
<p align="center">
    <kbd> <img width="1000" alt="workflow" src="https://drive.google.com/file/d/1kFyclaZkIEnsbc00FjDNjrLXY_7L3O__/view?usp=sharing"> </kbd> <br>
    Gambar 1 — Workflow Pembuatan Model
</p>
<br>
