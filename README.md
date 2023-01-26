# Cendekiawan-Sentiment-Analysis-Using-Naive-Bayes
ANALISIS SENTIMEN RESPON MASYARAKAT TERHADAP KNOWLEDGE SHARING CENDEKIAWAN DI TWITTER

#Algoritma: Naive Bayes

Langkah-langkah menggunakan model dan menjalankan aplikasi

# A. Penjelasan cycle model
1. Data Akuisisi
Data yang digunakan adalah data hasil Scraping menggunakan API twiter. Bisa dilihat untuk tutorial disini https://www.youtube.com/watch?v=ECPjw0w-qK0&t=28s

2. Data Exploration
Tahapan ini adalah dimana kita membaca data, Menganalisa, melihat struktur menggunakan beberapa tabel visualisasi seperti barchart, pie chart, dls.

3. modelling
tahapan ini adalah pembuatan struktur model dengan menggunakan struktur dari algoritma Naive Bayes. dengan hasil klasifikasi pada model yaitu klasifikasi hasil negatif, positif, dan netral

4. Evaluate
Evaluasi model menggunakan Convusion Metrix
1.	Nilai True Negatif (TN) merupakan jumlah data negatif yang terdeteksi dengan benar. 
2.	False Positive (FP) merupakan data negatif namun terdeteksi sebagai data positif. 
3.	True Positive (TP) merupakan data positif yang terdeteksi benar. 
4.	False Negatif (FN) merupakan kebalikan dari True Positive, sehingga data positif, namun terdeteksi sebagai data negatif. 
berdasarkan ke-4 nilai diatas diperoleh nilai Akurasi, Presisi, Recall, F-measure

5. Deployment
pada tahapan terakhir ini adalah deployment kedalam sebuah applikasi berbasis website dengan menggunakan Flask API


# B. Langkahlangkah menjalankan model dan aplikasi
1. git clone (code project)
2. Jika teman-teman belum membuat sebuah environment, disarankan untuk membuat environment terlebih dahulu. langkah pembuatannya cukup simpel, pertama buka anaconda prompt (jika pake anaconda prompt)

![image](https://user-images.githubusercontent.com/86903939/214845855-52226f39-1c6b-4aa5-a966-287103bac5ae.png)

pada base utama seperti pada gambar diatas, ketikkan conda create -n (nama environment) misalkan nama environmentnya adalah sentiment analisis
jika sudah enter saja, dan tunggu proses selesai..... jika sudah selesai teman-teman bisa cek environment yang sudah teman-teman buat apakah sudah berhasil atau belum dengan mengetikkan conda env list, seperti pada gambar 

![image](https://user-images.githubusercontent.com/86903939/214846485-a3ceedbc-06dc-476b-8790-e34ea0166c6a.png)

jika sudah, kalian bisa aktifkan environmentnya. misalkan saya aktifkan environment tf3-gpu dengan mengetikkan conda activate tf3-gpu

![image](https://user-images.githubusercontent.com/86903939/214846770-fe354481-10fd-496c-95e4-b1361c3240b3.png)

maka root base akan berganti dalam environment yang sudah kita aktifkan seperti pada gambar

![image](https://user-images.githubusercontent.com/86903939/214846943-f548b4bf-c1ae-4c82-9ff0-35e611aed4e2.png)

3. jika sudah pada environment yang sudah kalian buat, maka tinggal clone code project ini dengan menggunakan menuliskan perintah git clone (link code project)

4. Masuk ke folder project yang sudah teman-teman clone 
5. install requirements project dengan menuliskan perintah 

![image](https://user-images.githubusercontent.com/86903939/214849122-bc00121a-0c1c-4e89-a17b-3a37679a1d2b.png)

6. jalankan aplikasi dengan menuliskan perintah app.py

# Tampilan aplikasi:
1. Input text 

![inputan text](https://user-images.githubusercontent.com/86903939/214850827-cd9ac8d2-2d90-4d15-b909-39bd2b4000b4.jpeg)

2. Input File 

![image](https://user-images.githubusercontent.com/86903939/214852274-6c724d2e-452a-428b-bc75-960ed71734d4.png)

![output inpout file](https://user-images.githubusercontent.com/86903939/214851028-5e97cec2-9c19-45fe-be3f-439a406be254.jpeg)

![wordcloud](https://user-images.githubusercontent.com/86903939/214851113-c8211eb1-6a3e-4b8d-bef7-76aabd5a15e1.jpeg)

![graph analisis](https://user-images.githubusercontent.com/86903939/214851216-c8315a93-4720-4a70-8811-e655d49bec9c.jpeg)



