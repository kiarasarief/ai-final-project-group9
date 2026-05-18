# Tugas Akhir Kecerdasan Buatan (ENCE616029)
## Klasifikasi Citra Menggunakan Convolutional Neural Network (CNN)

Proyek ini merupakan tugas akhir untuk mata kuliah Kecerdasan Buatan, program studi Teknik Komputer, Universitas Indonesia. Proyek ini berfokus pada implementasi arsitektur Convolutional Neural Network (CNN) untuk menyelesaikan permasalahan klasifikasi citra.

### Anggota Kelompok
* Putri Kiara Salsabila Arief (2306250743)
* Neyla Shakira (2306250655)
* Vanesa Kayla Zahra (2306161901)
* Kharisma Aprilia (2306223244)

### Deskripsi Proyek
Proyek ini mengimplementasikan model deep learning berbasis CNN untuk mengenali dan mengklasifikasikan objek dari dataset citra. Model dikembangkan dengan memperhatikan parameter teknis seperti ukuran kernel, stride, padding, serta regularisasi menggunakan Batch Normalization dan Dropout untuk mencegah terjadinya overfitting. Proses eksperimen dan visualisasi training curve dicatat secara terstruktur untuk menganalisis performa akurasi dan loss model.

### Teknologi dan Framework
* Bahasa Pemrograman: Python
* Framework Deep Learning: TensorFlow / Keras atau PyTorch
* Library Pembantu: NumPy, Matplotlib, Scikit-Learn
* Lingkungan Pengembangan: Google Colab / GitHub

### Struktur Repositori
* `src/` : Berisi kode sumber utama untuk arsitektur model dan pipeline pelatihan.
* `notebooks/` : Google Colab notebook yang digunakan untuk eksperimen, eksplorasi data (EDA), dan pelatihan model.
* `docs/` : Laporan progress report dan laporan akhir proyek dalam format PDF.
* `metadata.json` : File metadata proyek sesuai dengan standar pengumpulan.

### Cara Menjalankan
1. Buka notebook yang tersedia di folder `notebooks/` menggunakan Google Colab.
2. Pastikan runtime telah diubah ke GPU (Runtime -> Change runtime type -> GPU).
3. Jalankan sel secara berurutan untuk memproses data, melatih model, dan melihat hasil evaluasi.
