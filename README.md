# Dokumentasi Project Pengolahan Citra Digital

## Latar Belakang
<br>Indonesia dikenal sebagai negara yang memiliki keanekaragaman hayati yang sangat tinggi, termasuk di dalamnya berbagai jenis tanaman herbal yang telah digunakan secara turun-temurun sebagai obat tradisional. Tanaman herbal seperti belimbing wuluh, jambu biji, lidah buaya, sirih, dan lainnya memiliki banyak manfaat untuk kesehatan, seperti meningkatkan daya tahan tubuh, menyembuhkan luka, hingga mencegah penyakit tertentu.

Namun, dalam praktiknya, masyarakat sering mengalami kesulitan dalam mengenali dan membedakan jenis tanaman herbal. Hal ini disebabkan oleh kemiripan bentuk daun antar tanaman, minimnya pengetahuan masyarakat, serta terbatasnya akses terhadap informasi yang akurat mengenai jenis dan manfaat dari tanaman herbal tersebut. Akibatnya, banyak orang yang tidak dapat memanfaatkan tanaman herbal secara optimal, bahkan bisa salah dalam penggunaannya yang dapat membahayakan kesehatan. Beradarkan [Artikel](https://ugm.ac.id/id/berita/22262-tips-menggunakan-tanaman-herbal/?utm_source=chatgpt.com) Dr. Djoko Santosa, beliau menyebutkan bahwa salah satu hal penting adalah “memastikan kebenaran dari tanaman yang hendak dikonsumsi, apakah tanaman tersebut adalah tanaman yang dimaksud atau hanya mirip saja”. Jika salah identifikasi, bisa jadi efeknya berlawanan atau berbahaya.

Seiring dengan berkembangnya teknologi kecerdasan buatan (Artificial Intelligence/AI), khususnya dalam bidang pengolahan citra digital (image processing), muncul peluang untuk membuat aplikasi yang dapat membantu proses identifikasi tanaman herbal secara otomatis, cepat, dan akurat. Dengan memanfaatkan deep learning, khususnya Convolutional Neural Network (CNN), sistem dapat dilatih menggunakan dataset gambar daun herbal untuk mengenali pola dan ciri khas dari setiap jenis tanaman.
<br>

## Arsitektur
<br>
Model pembelejaran ini menggunakan Arsitektur VGG16. Yaitu metode transfer learning yang selanjutnya akan ditambah dengan arsitektur layer baru pada bagian Fully Connected Layer yaitu output layer untuk menyesuaikan dengan jumlah kelas dataset yang diklasifikasikan.

## Dataset
Dataset yang di gunakan dalam model pembelajaran deep learning ini di dasari dari dataset yang dapat di akses pada link berikut
[Indonesian Herb Leaf Dataset 3500](https://data.mendeley.com/datasets/s82j8dh4rr/1)<br>
Terdapat sepuluh spesies tanaman yang terdapat dalam dataset ini, yaitu Averrhoa bilimbi (Blimbing Wuluh), Psidium guajava (Jambu Biji), Citrus Aurantiifolia (Jeruk Nipis), Ocimum Africanum (Kemangi), Aloe vera (Lidah Buaya), Artocarpus heterophyllus (Nangka), Pandanus Amaryllifolius (Pandan), Carica papaya (Pepaya), Apium graveolens (Seledri), dan Piper Betle (Sirih).
Total dataset terdiri dari 3500 gambar, di mana setiap spesies memiliki 350 gambar beresolusi tinggi. Folder diberi nama sesuai dengan nama dalam bahasa Indonesia.
Setiap gambar memiliki latar belakang putih, dengan format .jpg dan dimensi 1600 x 1200 piksel.


## Deployment
Deployment pada aplikasi ini di upload dalam Streamlit karena arsitektur VGG16 atau transfer learning tidak cocok untuk penggunaan model secara realtime. Saya sarankan jika ingin di rubah dalam mobile maka gunakan Arsitektur MobileNet. 
Untuk aplikasi bisa di klik di link dibawah ini
<br>
[Herbal Leaf Prediction](https://herbal-leaf.streamlit.app/)
