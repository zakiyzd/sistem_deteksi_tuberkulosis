  Sistem Deteksi Tuberkulosis dari Citra X-Ray Dada

  Gambaran Umum Sistem

  Proyek ini dikembangkan untuk membantu dalam deteksi dan klasifikasi tuberkulosis (TB) menggunakan citra X-ray dada. Sistem ini menyediakan solusi end-to-end, mulai dari pemrosesan data mentah
  X-ray, pelatihan model kecerdasan buatan, hingga penyediaan antarmuka web yang mudah digunakan untuk membantu diagnosis. Tujuan utama adalah untuk memberikan alat bantu yang efisien dan dapat   
  diandalkan kepada profesional medis dalam mendiagnosis TB.

  Cara Kerja Sistem

  Sistem ini beroperasi dalam beberapa fase utama:

   1. Pemrosesan Data: Citra X-ray dada dan metadata yang relevan dikumpulkan dan diproses terlebih dahulu. Ini melibatkan langkah-langkah seperti normalisasi, penskalaan, dan augmentasi data
      untuk memastikan kualitas dan kuantitas data yang memadai untuk pelatihan model.
   2. Pelatihan Model *Deep Learning*: Sebuah model Deep Learning, khususnya arsitektur ResNet, dilatih menggunakan set data X-ray dada yang telah diproses. Model ini belajar untuk
      mengidentifikasi pola dan fitur-fitur spesifik dalam citra yang menunjukkan keberadaan atau ketiadaan tuberkulosis. Proses pelatihan ini diulang dan dioptimalkan untuk mencapai performa
      deteksi yang terbaik.
   3. Aplikasi Web Interaktif: Setelah model dilatih dan divalidasi, ia diintegrasikan ke dalam aplikasi web berbasis Flask. Pengguna dapat mengunggah citra X-ray dada melalui antarmuka ini, dan
      aplikasi kemudian akan menggunakan model terlatih untuk menganalisis citra tersebut dan memberikan prediksi mengenai status TB. Ini memungkinkan penggunaan model secara praktis dalam
      lingkungan yang ramah pengguna.

  Teknologi yang Digunakan

  Proyek ini memanfaatkan berbagai teknologi untuk mencapai tujuannya:

   * Pemrograman: Python
   * Kerangka Kerja Web: Flask (untuk aplikasi web frontend dan backend)
   * Pembelajaran Mendalam (*Deep Learning*): TensorFlow dan Keras (untuk pengembangan dan pelatihan model ResNet)
   * Manajemen Data: Pandas dan NumPy (untuk pemrosesan dan analisis data)
   * Penyimpanan Model: File .h5 (untuk menyimpan model Deep Learning yang telah dilatih)
   * Pengembangan Eksperimen: Jupyter Notebooks (untuk eksplorasi data, prototipe model, dan analisis)
   * Lingkungan Pengembangan: Visual Studio Code ( .vscode/ )
   * Manajemen Paket (Diduga): Conda
