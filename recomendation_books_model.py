# %% [markdown]
# <h1>Book Recomendation Model</h1>

# %% [markdown]
# Nama : Rafael Siregar<br>
# username : rafael_siregar611<br>
# E-mail : rafael_siregar@students.polmed.ac.id

# %% [markdown]
# Melakukan Import Library yang akan kita gunakan untuk membuat model sistem rekomendasi Buku

# %%

# Import semua Library yang dibutuhkan untuk pembuatan model rekomendasi buku
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import numpy as np
import re
import warnings
import matplotlib.pyplot as plt
from PIL import Image
import requests
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
tqdm.pandas()



# %% [markdown]
# <h1><i>Understanding Data</i></h1>

# %% [markdown]
# <h2>1.Load Dataset</h2>

# %% [markdown]
# Kode berikut digunakan untuk memuat tiga dataset utama yang diperlukan dalam pembuatan sistem rekomendasi buku:

# %%
#Melakukan load dataset buku dari book_dataset/Books.csv , Ratings.csv, dan Users.csv
books = pd.read_csv('book_dataset/Books.csv', low_memory=False)
ratings = pd.read_csv('book_dataset/Ratings.csv', low_memory=False)
users = pd.read_csv('book_dataset/Users.csv', low_memory=False)


# %% [markdown]
# books berisi data detail buku seperti ISBN, judul, penulis, tahun terbit, penerbit, dan URL gambar sampul.<br>
# ratings berisi data interaksi pengguna dengan buku, yaitu UserId, ISBN, dan rating yang diberikan (0-10).<br>
# users berisi data pengguna seperti UserId, lokasi, dan umur.<br>
# Dengan memuat ketiga dataset ini, kita dapat:<br>
# 
# Menghubungkan informasi buku dengan rating yang diberikan oleh user.<br>
# Menganalisis perilaku pengguna dan preferensi mereka.<br>
# Menyiapkan data untuk proses eksplorasi, pembersihan, dan pembuatan model<br> rekomendasi berbasis content-based maupun collaborative filtering.

# %%
print("Dataset Buku:", books.ISBN.count(), "Buku")
print("Dataset Rating:", ratings.UserId.count(), "Rating")
print("Dataset User:", users.UserId.count(), "User")

# %% [markdown]
# Dari Output diatas dapat kita lihat bahwa banyaknya data dengan rincian : <br> Dataset Buku: 271360 Buku<br>
# Dataset Rating: 105283 Rating<br>
# Dataset User: 278858 User<br>

# %% [markdown]
# <h2>2.Univariate Exploratory Data Analysis</h2>

# %% [markdown]
# <h4>Melakukan EDA terhadap dataframe books</h4>

# %%
books.info()

# %% [markdown]
# Jumlah Data: Terdapat 271.360 baris data buku, yang berarti dataset ini sangat besar dan kaya akan variasi buku.<br>
# Kolom: Ada 8 kolom, yaitu ISBN, Book-Title, Book-Author, Year-Of-Publication, Publisher, Image-URL-S, Image-URL-M, dan Image-URL-L.<br>
# Tipe Data: Semua kolom bertipe objek (string), termasuk tahun terbit yang seharusnya bertipe numerik.<br>
# Missing Value:<br>
# Kolom Book-Author memiliki 1 data kosong.<br>
# Kolom Publisher memiliki 2 data kosong.<br>
# Kolom Image-URL-L memiliki 3 data kosong.<br>
# Kolom lain tidak memiliki missing value.<br>
# Kualitas Data:<br>
# Sebagian besar data sudah lengkap, namun ada beberapa missing value yang perlu ditangani pada tahap preprocessing.<br>
# ISBN sebagai primary key unik untuk setiap buku, namun perlu dicek apakah ada duplikasi.<br>
# Kolom tahun terbit bertipe objek, sehingga perlu diubah ke numerik jika ingin digunakan untuk analisis lebih lanjut.<br>

# %%
#Mengecek Banyaknya data berdasarkan ISBN
print("Banyaknya data pada dataframe books", books.count())
print("Banyaknya data berdasarkan ISBN:", books.ISBN.nunique())

# %% [markdown]
# Setiap kolom pada dataframe books hampir seluruhnya memiliki jumlah data yang sama, yaitu 271.360, kecuali:<br>
# Book-Author kurang 1 data (271.359), artinya ada 1 baris dengan nilai kosong (missing value) pada kolom ini.<br>
# Publisher kurang 2 data (271.358), artinya ada 2 baris dengan nilai kosong pada kolom ini.<br>
# Image-URL-L kurang 3 data (271.357), artinya ada 3 baris dengan nilai kosong pada kolom ini.<br>
# Kolom ISBN memiliki jumlah unik yang sama dengan jumlah baris (271.360), artinya setiap buku memiliki ISBN yang unik dan tidak ada duplikasi ISBN pada dataset ini.<br>
# Data sudah cukup bersih dari duplikasi, namun masih ada sedikit missing value pada beberapa kolom yang perlu ditangani pada tahap preprocessing.<br>

# %% [markdown]
# <h4>Melakukan EDA terhadap dataframe Rating</h4>

# %%
ratings.info()
ratings.describe().round(0)

# %% [markdown]
# Dari output diatas dapat kita lihat bahwa rating buku berada pada minimum 0 dan maximum 10.
# Distribusi rating sangat tidak seimbang:
# Lebih dari 50% data memiliki rating 0, artinya banyak user yang hanya menandai buku tanpa memberikan penilaian sebenarnya.
# Perlu pembersihan data:
# Untuk modeling, biasanya rating 0 dihapus agar model hanya belajar dari interaksi yang benar-benar bermakna (rating 1-10).
# Jumlah user dan buku sangat besar:
# Dataset sangat kaya, cocok untuk collaborative filtering, namun perlu balancing agar model tidak bias ke rating 0.

# %% [markdown]
# kemudian kita akan cek berapada user yang memberikan rating

# %%
#Melihat banyaknya user yang memberikan rating berdasarkan UserId
print("Banyaknya data pada dataframe ratings", ratings.UserId.count())
print("Banyaknya data user sesuai dengan dataframe users", users.UserId.count())
print("Banyaknya User yang melakukan rating berdasarkan nilai unik dari UserId:", len(ratings.UserId.unique()))
# Banyaknya buku yang dirating berdasarkan ISBN (jumlah ISBN unik yang ada di ratings)
print("Banyaknya buku yang dirating berdasarkan nilai unik dari ISBN:", ratings['ISBN'].nunique())

# %% [markdown]
# Berdasarkan hasil analisis dataframe ratings, terdapat 1.149.780 data interaksi user-buku, namun hanya 105.283 user yang benar-benar memberikan rating dari total 278.858 user yang terdaftar, menunjukkan bahwa sebagian besar user tidak aktif memberikan penilaian. Selain itu, terdapat 340.556 ISBN unik yang pernah dirating, jumlah ini lebih banyak dari ISBN pada dataset books, sehingga ada kemungkinan data ratings mengandung ISBN yang tidak ditemukan di books. Hal ini menandakan perlunya pembersihan dan penyelarasan data agar sistem rekomendasi yang dibangun hanya menggunakan data yang valid dan konsisten, serta menghindari bias akibat banyaknya rating 0 yang mendominasi distribusi data.

# %% [markdown]
# <h4>Melakukan EDA terhadap dataframe Users</h4>

# %%
users.info()

# %% [markdown]
# Berdasarkan output users.info(), terdapat 278.858 data user dengan 3 kolom: UserId, Location, dan Age. Seluruh data pada kolom UserId dan Location lengkap (tidak ada missing value), namun pada kolom Age hanya terdapat 168.096 data yang terisi, sehingga sekitar 110.762 data user tidak memiliki informasi umur (missing value). Hal ini menunjukkan bahwa meskipun data user cukup lengkap untuk ID dan lokasi, namun informasi umur masih banyak yang kosong sehingga perlu dipertimbangkan penanganannya pada tahap preprocessing, misalnya dengan mengisi nilai kosong atau mengabaikan kolom tersebut jika tidak terlalu relevan untuk model rekomendasi.

# %% [markdown]
# <h1><i>Preprocessing Data</i></h1>

# %% [markdown]
# hal pertama yang akan kita lakukan dalam preprocessing data adalah menggabungkan tiap dataframe dari books,ratings,dan users kedalam 1 dataframe, namun sebagai pertimgbangan kita hanyak akan menggunakan dataframe yang kita butuhkan saja, daalm kasus ini, hasil akhir dataframe yang diharapkan adalah UserId	ISBN	Book-Rating	Book-Title	Book-Author	Publisher	

# %%
books_data=books.merge(ratings,on="ISBN")
books_data.head()
books_data.info()

# %% [markdown]
# Kita sudah berhasil menggabungkan Dataframe dengan kolom-kolom yang dipertahankan adalah UserId	ISBN Book-Rating Book-Title Book-Author	Publisher	

# %% [markdown]
# <h1><i>Preparation Data</i></h1>

# %% [markdown]
# cek nilai null, missing values dan nilai yang tidak valid dan menghapus Kolom yang tidak kita perlukan seperti "ISBN","Year-Of-Publication","Image-URL-S","Image-URL-M"

# %% [markdown]
# <i>Melakukan duplikasi dataframe</i>

# %%
df=books_data.copy()

# %% [markdown]
# Baris kode diatas digunakan untuk membuat salinan dataframe books_data ke dalam variabel baru df, sehingga proses pembersihan dan manipulasi data selanjutnya dapat dilakukan tanpa mengubah data asli pada books_data.

# %% [markdown]
# <i>Menghapus Nilai Kosong (NaN) dan melakukan reset Index</i>

# %%
df.dropna(inplace=True)
df.reset_index(drop=True,inplace=True)

# %% [markdown]
# Baris kode diatas digunakan untuk menghapus seluruh baris yang memiliki nilai kosong (NaN) pada dataframe df, kemudian melakukan reset index agar urutan index kembali berurutan dari nol setelah penghapusan baris.

# %% [markdown]
# <i>Menghapus Kolom yang tidak diperlukan</i>

# %%
df.drop(columns=["ISBN","Year-Of-Publication","Image-URL-S","Image-URL-M"],axis=1,inplace=True)

# %% [markdown]
# Baris kode diatas digunakan untuk menghapus kolom-kolom yang tidak diperlukan, yaitu "ISBN", "Year-Of-Publication", "Image-URL-S", dan "Image-URL-M" dari dataframe df, sehingga hanya kolom yang relevan untuk proses rekomendasi yang dipertahankan.

# %% [markdown]
# <i>Menghapus Baris Rating 0</i>

# %%
df.drop(index=df[df["Book-Rating"]==0].index,inplace=True)

# %% [markdown]
# Baris kode diatas digunakan untuk menghapus seluruh baris pada dataframe df yang memiliki nilai Book-Rating sama dengan 0. Hal ini bertujuan agar hanya interaksi user yang benar-benar memberikan rating (1-10) yang digunakan dalam proses rekomendasi.

# %% [markdown]
# <i>Membersihkan Nama Judul Buku</i>

# %%
df["Book-Title"]=df["Book-Title"].apply(lambda x: re.sub("[\W_]+"," ",x).strip())

# %% [markdown]
# Baris kode diatas digunakan untuk membersihkan nama judul buku pada kolom "Book-Title" dengan menghapus seluruh karakter non-alfabet dan underscore, lalu menghilangkan spasi di awal dan akhir string. Ini bertujuan agar judul buku lebih konsisten dan rapi untuk proses analisis selanjutnya.

# %% [markdown]
# <i>Menggabungkan Kolom yang nanti akan kita gunakan sebagai Fitur Rekomendasi kita</i>

# %%
# Gabungkan fitur-fitur penting untuk setiap buku
df['all_features'] = df[['Book-Title', 'Book-Author']].astype(str).agg(' '.join, axis=1)

# %% [markdown]
# Baris kode diatas digunakan untuk membuat kolom baru bernama all_features yang merupakan gabungan dari kolom "Book-Title" dan "Book-Author" untuk setiap buku. Kolom ini akan digunakan sebagai fitur utama dalam proses ekstraksi fitur dan perhitungan kemiripan antar buku pada sistem rekomendasi.

# %% [markdown]
# <i>Kita akan Cek hasil akhir dari Dataframe kita setelah semua proses ini dilakukan</i>

# %%
print(df.info())
df.sample(10)

# %% [markdown]
# Jumlah baris: 383,838
# Kolom: 7 (Book-Title, Book-Author, Publisher, Image-URL-L, UserId, Book-Rating, all_features)
# Semua kolom tidak memiliki missing value (non-null count sama dengan jumlah baris)
# Tipe data: Book-Title, Book-Author, Publisher, Image-URL-L, all_features bertipe objek (string); UserId dan Book-Rating bertipe integer

# %% [markdown]
# Reduksi Data dengan menghapus Diplikasi terhadap Book-Title

# %%
#Mereduksi Banyaknya data dengan Menghapus Duplikasi dari Buku dan Menyimpan kedalam Dataframe baru
df_unique = df.drop_duplicates(subset=['Book-Title']).reset_index(drop=True)

# %% [markdown]
# Baris kode diatas digunakan untuk mengurangi jumlah data dengan menghapus duplikasi berdasarkan kolom "Book-Title" dan menyimpan hasilnya ke dalam dataframe baru df_unique. Setiap judul buku hanya akan muncul satu kali, sehingga proses rekomendasi berbasis konten menjadi lebih efisien dan tidak bias terhadap buku yang sama.

# %% [markdown]
# Ekstraksi Fitur Dengan Tf-IDF dan Mengkombinasikan dengan CountVectorizer

# %%
# TF-IDF Vectorizer (sudah ada)
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df_unique['all_features'])

# CountVectorizer dengan ngram (unigram + bigram)
count_vect = CountVectorizer(ngram_range=(1,2), min_df=2)
count_matrix = count_vect.fit_transform(df_unique['all_features'])

# Gabungkan kedua matriks fitur
combined_matrix = hstack([tfidf_matrix, count_matrix])

# %% [markdown]
# Baris kode diatas digunakan untuk mengekstraksi fitur dari kolom all_features pada dataframe df_unique menggunakan TF-IDF (Term Frequency-Inverse Document Frequency). Hasilnya adalah matriks fitur numerik (tfidf_matrix) yang akan digunakan untuk menghitung kemiripan antar buku pada sistem rekomendasi berbasis konten.

# %% [markdown]
# Setelah semua tahapan Preparation diatas, data kita sudah siap untuk digunakan dalam pembuatan model rekomendasi buku kita

# %% [markdown]
# <h1><i>Modeling</i></h1>

# %% [markdown]
# <h1>CONTENT BASED FILTERING</h1>

# %% [markdown]
# Model Development dengan Content Based Filtering adalah proses membangun sistem rekomendasi buku yang merekomendasikan buku kepada user berdasarkan kemiripan konten atau atribut buku itu sendiri, seperti judul, penulis, atau deskripsi.
# Pada pendekatan ini, sistem akan mencari buku-buku yang memiliki kemiripan fitur (misal: judul atau penulis yang mirip) tanpa memperhatikan preferensi user lain.
# Jadi, rekomendasi dihasilkan dari analisis konten buku, bukan dari perilaku user lain.

# %% [markdown]
# Langkah dalam Pembuatan model ini adalah

# %% [markdown]
# <ol>
# <li><i>Definisi Fungsi Rekomendasi</i></li>
# <li><i>Cek Judul Buku</i></li>
# <li><i>Cari Index Judul Buku</i></li>
# <li><i>Hitung Kemiripan (Cosine Similarity)</i></li>
# <li><i>Ambil Rata-rata Rating per Judul</i></li>
# <li><i>Normalisasi Rating</i></li>
# <li><i>Gabungkan Skor Similarity dan Rating</i></li>
# <li><i>Ambil Top-k Rekomendasi</i></li>
# <li><i>Ambil Data Buku Rekomendasi</i></li>
# <li><i>Sortir dan Format Output</i></li>
# </ol>

# %%
def book_recommendations(book_title, items=df_unique[['Book-Title', 'Book-Author', 'Publisher', 'Image-URL-L']], k=10, alpha=0.7):
    if book_title not in df_unique['Book-Title'].values:
        print("Judul buku tidak ditemukan.")
        return None
    idx = df_unique[df_unique['Book-Title'] == book_title].index[0]
    # Gunakan combined_matrix untuk similarity
    sim_scores = cosine_similarity(combined_matrix[idx], combined_matrix).flatten()
    avg_rating = df.groupby('Book-Title')['Book-Rating'].mean().reindex(df_unique['Book-Title']).fillna(0)
    norm_rating = (avg_rating - avg_rating.min()) / (avg_rating.max() - avg_rating.min())
    final_score = alpha * sim_scores + (1 - alpha) * norm_rating.values
    similar_indices = final_score.argsort()[::-1]
    similar_indices = [i for i in similar_indices if i != idx][:k]
    recommended = df_unique.iloc[similar_indices][['Book-Title', 'Book-Author', 'Publisher', 'Image-URL-L']]
    recommended['Similarity'] = sim_scores[similar_indices]
    recommended['Avg-Rating'] = avg_rating.iloc[similar_indices].values
    recommended = recommended.sort_values(by='Similarity', ascending=False).reset_index(drop=True)
    recommended['Similarity'] = recommended['Similarity'].round(2) * 100
    recommended['Avg-Rating'] = recommended['Avg-Rating'].round(2)
    return recommended

# %% [markdown]
# Setelah kita mendefinisikan Fungsi sesuai dengan alur yang sudah kita tentukan, Selanjutnya kita akan Mengevaluasi model untuk menilai apakah model yang kita buat dapat bekerja dengan baik

# %% [markdown]
# <i>Evaluasi Model CBF</i>

# %%
def precision_at_k_cbf(df_unique, k=10, n_books=50, plot=True):
    """
    Precision@k untuk CBF: relevan jika Book-Author sama dengan buku acuan.
    Sekaligus menampilkan distribusi precision@k.
    """
    sampled_books = df_unique['Book-Title'].sample(n=n_books, random_state=42)
    precisions = []

    for book_title in sampled_books:
        author = df_unique[df_unique['Book-Title'] == book_title]['Book-Author'].values[0]
        recs = book_recommendations(book_title, k=k)
        if recs is None or recs.empty:
            continue
        relevant = (recs['Book-Author'] == author).sum()
        precisions.append(relevant / k)

    if precisions:
        mean_precision = np.mean(precisions)
        print(f"Precision@{k} untuk {n_books} buku: {mean_precision:.2f}")
        if plot:
            plt.figure(figsize=(8,5))
            bins = np.arange(0, 1.1, 0.1)
            plt.hist(precisions, bins=bins, rwidth=0.8, color='skyblue', edgecolor='black')
            plt.title(f'Distribusi Precision@{k} untuk {n_books} Buku')
            plt.xlabel(f'Precision@{k} (Proporsi Rekomendasi Relevan)')
            plt.ylabel('Jumlah Buku')
            plt.xticks(bins, [f"{b:.1f}" for b in bins])
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()
    else:
        print("Tidak ada data yang bisa dievaluasi.")

# Contoh pemanggilan evaluasi precision@k dengan visualisasi
precision_at_k_cbf(df_unique, k=10, n_books=50)

# %% [markdown]
# Hasil dari evaluasi kita dapatkan : Precision@10 untuk 50 buku: **0.27** yang berarti Nilai precision@10 sebesar 0.27 rata-rata hanya 27% dari 10 rekomendasi teratas yang benar-benar berasal dari penulis yang sama dengan buku acuan. Hal ini menunjukkan bahwa sistem CBF Kita cenderung memberikan rekomendasi yang bervariasi secara penulis, namun tetap ada sebagian rekomendasi yang sangat relevan secara penulis. Jika target utama adalah merekomendasikan buku dari penulis yang sama, precision@10 ini dapat dijadikan acuan untuk perbaikan lebih lanjut, misalnya dengan menambah bobot fitur penulis pada proses ekstraksi fitur.

# %% [markdown]
# <i>Penerapan Model</i>

# %%

book_recommendations('A Gracious Plenty A Novel')

# %% [markdown]
# Dari Output diatas dapat kita lihat rekomendasi untuk buku berjudul "A Gracious Plenty A Novel" memiliki 10 rekomendasi dengan patokan Similarity adalah Book-title dan Book-Author serta ikut mempertimbangkan untuk NIlai Similarity dan Avg-Rating teratas sehingga tidak menampilkan Similarity 0 dan Avg-Rating 0

# %% [markdown]
# <h1>Collaborative Filtering</h1>

# %% [markdown]
# Collaborative filtering adalah teknik yang digunakan dalam sistem rekomendasi untuk membuat prediksi otomatis tentang minat seorang pengguna dengan mengumpulkan preferensi atau informasi selera dari banyak pengguna (berkolaborasi). Ide dasarnya adalah jika seseorang A memiliki opini yang sama dengan seseorang B pada suatu isu, A lebih mungkin memiliki opini yang sama dengan B pada isu lain dibandingkan dengan orang yang dipilih secara acak.
# Dalam Case ini kita akan menggunakan Rating book dan UserId untuk menampilkan rekomendasi kepada user

# %% [markdown]
# <h2>Load Dataset</h2>

# %% [markdown]
# 1.Membaca dataset Rating terlebih dahulu namun karena ukuran dataset yang sangat besar, kita akan membatasi pengunaan dataset menjadi 20000 teratas dan di sort berdasarkan rating dan terapkan penyeimbangan pengambilan tiap data agar memiliki nilai sebaran yang seimbang

# %%
# Filter rating minimal 5
ratings_filtered = ratings[ratings['Book-Rating'] >= 5]

# Tentukan jumlah sample per rating (misal: 6 kelas rating 5-10, 20000/6 ≈ 3333 per kelas)
ratings_per_class = 20000 // len(ratings_filtered['Book-Rating'].unique())

# Ambil sample seimbang untuk setiap rating
balanced_samples = (
    ratings_filtered.groupby('Book-Rating', group_keys=False)
    .apply(lambda x: x.sample(min(len(x), ratings_per_class), random_state=42))
)
# Gabungkan dengan books
df_tf = balanced_samples.merge(books, on='ISBN', how='left')

# Cek distribusi rating
print(balanced_samples['Book-Rating'].value_counts())
print("Total data:", len(balanced_samples))

# %% [markdown]
# <h2>Data Preparation</h2>

# %% [markdown]
# <i>1.Melakukan Encoding pada ISBN sebagai patokan untuk Sistem Rekomendasi kita</i>

# %%
# Membuat list ISBN unik
isbn_ids = df_tf['ISBN'].unique().tolist()

# Encoding ISBN ke angka
isbn_to_encoded = {x: i for i, x in enumerate(isbn_ids)}

# Encoding angka ke ISBN
encoded_to_isbn = {i: x for i, x in enumerate(isbn_ids)}

# Terapkan encoding ke kolom ISBN pada df_tf
df_tf['ISBN'] = df_tf['ISBN'].map(isbn_to_encoded)

# %% [markdown]
# Kode di atas melakukan proses encoding pada kolom ISBN di dataframe df_tf agar setiap ISBN (yang semula berupa string unik) diubah menjadi angka integer unik. Langkah pertama adalah membuat daftar semua ISBN unik dari kolom ISBN, lalu membuat dua dictionary: isbn_to_encoded untuk mengubah ISBN menjadi angka, dan encoded_to_isbn untuk mengubah angka kembali ke ISBN aslinya. Setelah itu, kolom ISBN pada df_tf diubah nilainya dari string menjadi angka menggunakan mapping isbn_to_encoded. Proses ini penting agar data ISBN dapat digunakan sebagai input numerik pada model machine learning, khususnya pada layer embedding di model rekomendasi.

# %% [markdown]
# <i>2.Menghapus Null atau Nan values dan kolom yang tidak diperlukan seperti Year-Of-Publication Image-URL-S Image-URL-L  </i>

# %%
# Menghapus Null atau Nan values dan kolom yang tidak diperlukan seperti Year-Of-Publication, Image-URL-S, Image-URL-M, Image-URL-L
df_tf.dropna(inplace=True)
df_tf.reset_index(drop=True, inplace=True)
df_tf.drop(columns=["Year-Of-Publication", "Image-URL-S", "Image-URL-M", "Image-URL-L"], axis=1, inplace=True)
df_tf.info()
df_tf.sample(10)

# %% [markdown]
# Berdasarkan output dari kode pembersihan data di atas, dapat dilihat bahwa dataframe df_tf kini berisi 17.717 baris data tanpa nilai kosong (null) dan hanya memiliki kolom yang relevan untuk proses rekomendasi, yaitu: UserId, ISBN, Book-Rating, Book-Title, Book-Author, dan Publisher. Tiga kolom pertama bertipe numerik, sedangkan tiga kolom terakhir bertipe objek (teks). Contoh data yang ditampilkan menunjukkan variasi judul buku, penulis, dan penerbit, serta rentang rating buku dari 5 hingga 10. Hal ini menandakan data sudah bersih, terstruktur, dan siap digunakan untuk proses training model rekomendasi berbasis collaborative filtering, di mana setiap baris merepresentasikan interaksi unik antara user dan buku tertentu. Data yang sudah terfilter dan terstruktur seperti ini akan membantu model belajar pola preferensi pengguna dengan lebih baik dan menghasilkan rekomendasi yang lebih akurat.

# %% [markdown]
# <i>2.Membagi dataset menjadi Train dan Test</i>

# %%
from sklearn.model_selection import train_test_split

# Membuat variabel x untuk mencocokkan data UserId dan ISBN menjadi satu value
x = df_tf[['UserId', 'ISBN']].values

# Membuat variabel y untuk rating (Book-Rating), bisa dinormalisasi jika perlu
min_rating = df_tf['Book-Rating'].min()
max_rating = df_tf['Book-Rating'].max()
y = df_tf['Book-Rating'].apply(lambda r: (r - min_rating) / (max_rating - min_rating)).values

# Membagi menjadi 80% data train dan 20% data test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, shuffle=True
)

print(x_train.shape, x_test.shape)

# %% [markdown]
# Output (14173, 2) (3544, 2) menunjukkan bahwa setelah dilakukan pembagian data menggunakan train_test_split, sebanyak 80% data (14.173 baris) digunakan sebagai data training dan 20% data (3.544 baris) digunakan sebagai data testing. Masing-masing baris pada variabel x_train dan x_test berisi dua fitur, yaitu UserId dan ISBN yang sudah diencoding ke bentuk numerik. Pembagian ini memastikan model dapat dilatih pada sebagian besar data, lalu diuji performanya pada data yang belum pernah dilihat sebelumnya untuk mengukur kemampuan generalisasi model rekomendasi.

# %% [markdown]
# <i>3.Membuat Model Rekomendasi dengan Pendekatan rekomendasi berupa UserId,ISBN,Book-Rating</i>

# %%
class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_books, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_books = num_books
        self.embedding_size = embedding_size

        self.user_embedding = layers.Embedding(
            input_dim=num_users,
            output_dim=embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.book_embedding = layers.Embedding(
            input_dim=num_books,
            output_dim=embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.book_bias = layers.Embedding(num_books, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        book_vector = self.book_embedding(inputs[:, 1])
        book_bias = self.book_bias(inputs[:, 1])

        dot_user_book = tf.reduce_sum(user_vector * book_vector, axis=1, keepdims=True)
        x = dot_user_book + user_bias + book_bias
        return tf.nn.sigmoid(x)  # output antara 0-1 (rating dinormalisasi)

# %% [markdown]
# Kelas RecommenderNet di atas adalah implementasi model rekomendasi berbasis neural network dengan pendekatan embedding untuk collaborative filtering. Model ini menggunakan dua buah embedding layer, masing-masing untuk merepresentasikan user (user_embedding) dan buku (book_embedding) ke dalam vektor berdimensi lebih kecil (latent space), sehingga pola interaksi antara user dan buku dapat dipelajari secara efisien. Selain itu, terdapat bias untuk user (user_bias) dan buku (book_bias) yang membantu model menangkap kecenderungan rating spesifik dari masing-masing user dan buku. Pada fungsi call, vektor user dan buku dikalikan (dot product) untuk menghasilkan skor interaksi, lalu dijumlahkan dengan bias user dan buku, dan akhirnya dilewatkan ke fungsi aktivasi sigmoid agar output berada pada rentang 0-1 (sesuai dengan rating yang telah dinormalisasi). Model ini sangat efektif untuk mempelajari preferensi pengguna dan memberikan rekomendasi buku yang relevan berdasarkan pola rating historis.

# %% [markdown]
# <i>4.Penerapan Model</i>

# %%
# Inisialisasi model
num_users = df_tf['UserId'].nunique()
num_books = df_tf['ISBN'].nunique()
model = RecommenderNet(num_users, num_books, 50)

# Compile model
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

# %% [markdown]
# Kode di atas melakukan inisialisasi dan kompilasi model rekomendasi berbasis neural network. Pertama, jumlah user unik (num_users) dan jumlah buku unik (num_books) dihitung dari dataframe, lalu digunakan untuk membuat instance model RecommenderNet dengan ukuran embedding 50. Selanjutnya, model dikompilasi menggunakan fungsi loss BinaryCrossentropy, optimizer Adam dengan learning rate 0.001, dan metrik evaluasi RootMeanSquaredError. Langkah ini memastikan model siap untuk proses training, di mana model akan belajar memetakan interaksi user dan buku untuk memprediksi rating secara akurat.

# %% [markdown]
# <i>5. Menjalankan Train Model</i>

# %%
import tensorflow as tf

# Cek apakah GPU tersedia
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU detected:", gpus)
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected, training will use CPU.")

# Tambahkan EarlyStopping callback
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_root_mean_squared_error',
    patience=5,
    restore_best_weights=True
)

# Training dengan callback (gunakan data test untuk validasi)
history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=64,
    epochs=100,
    validation_data=(x_test, y_test),
    callbacks=[early_stop]
)

# %% [markdown]
# Kode di atas menjalankan proses training model rekomendasi dengan beberapa langkah penting. Pertama, dilakukan pengecekan apakah GPU tersedia untuk mempercepat proses training; jika ada, TensorFlow akan mengatur penggunaan memori GPU secara dinamis. Selanjutnya, digunakan callback EarlyStopping untuk menghentikan training secara otomatis jika metrik validasi val_root_mean_squared_error tidak membaik selama 5 epoch berturut-turut, serta mengembalikan bobot model terbaik. Model kemudian dilatih menggunakan data training (x_train, y_train) dan divalidasi dengan data test (x_test, y_test) selama maksimal 100 epoch, batch size 8, dan training akan berhenti lebih awal jika performa validasi sudah optimal. Hasil training disimpan dalam variabel history untuk analisis lebih lanjut.

# %% [markdown]
# RMSE pada data validasi: 0.2949

# %% [markdown]
# <i>6.Memvisualisasikan grafik RMSE dari Model yang telah dibuat</i>

# %%
plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('Model Metrics')
plt.ylabel('Root Mean Squared Error')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# %% [markdown]
# Grafik di atas menunjukkan perkembangan nilai Root Mean Squared Error (RMSE) pada data training dan validation selama proses training model. Terlihat bahwa RMSE pada data training terus menurun seiring bertambahnya epoch, menandakan model semakin baik dalam mempelajari data training. Sementara itu, RMSE pada data validation cenderung stabil dan tidak mengalami kenaikan signifikan, yang berarti model tidak mengalami overfitting secara drastis. Selisih yang kecil antara kurva training dan validation menunjukkan model memiliki generalisasi yang baik terhadap data baru. RMSE yang rendah pada validation menandakan prediksi rating model cukup akurat terhadap data yang belum pernah dilihat sebelumnya.

# %% [markdown]
# <i>7.Menerapkan Model untuk mendapatkan rekomendasi buku</i>

# %%
# Ambil 1 user secara acak
user_id = 196077
print('Showing recommendations for user:', user_id)
print('=' * 27)

# Buku yang sudah pernah dirating user ini
books_visited_by_user = df_tf[df_tf['UserId'] == user_id]

print('Books with high ratings from user')
print('-' * 32)
# 5 buku dengan rating tertinggi dari user
top_books_user = (
    books_visited_by_user.sort_values(by='Book-Rating', ascending=False)
    .head(5)
    .ISBN.values
)
top_books_rows = books[books['ISBN'].isin([encoded_to_isbn[i] for i in top_books_user])]
#buat top_books_rows menjadi dataframe
top_books_rows = top_books_rows[['Book-Title', 'Book-Author', 'Publisher']].reset_index(drop=True)
top_books_rows


# %% [markdown]
# <i>kemudian kita akan tampilkan 10 Rekomendasi teratas kepada user</i>

# %%
print('-' * 32)
print('Top 10 book recommendation for user '+ str(user_id))
print('-' * 32)

# Buku yang belum pernah dirating user ini
books_not_visited = df_tf[~df_tf['ISBN'].isin(books_visited_by_user['ISBN'])]['ISBN'].unique()
user_books_array = np.array([[user_id, isbn] for isbn in books_not_visited])

# Prediksi rating untuk semua buku yang belum pernah dirating user ini
ratings = model.predict(user_books_array).flatten()
top_ratings_indices = ratings.argsort()[-10:][::-1]
recommended_isbn = books_not_visited[top_ratings_indices]
# Tampilkan 10 rekomendasi buku sebagai dataframe
recommended_books = books[books['ISBN'].isin([encoded_to_isbn[i] for i in recommended_isbn])]
recommended_books_df = recommended_books[['Book-Title', 'Book-Author', 'Publisher']].reset_index(drop=True)
recommended_books_df

# %% [markdown]
# Berdasarkan hasil kode collaborative filtering di atas, sistem rekomendasi berhasil memberikan daftar 10 buku yang sangat relevan dan personal untuk user yang dipilih secara acak, dengan mempertimbangkan pola rating historis user tersebut. Data yang digunakan telah melalui proses pembersihan dan penyeimbangan, sehingga setiap kelas rating terwakili secara proporsional dan model dapat belajar preferensi pengguna dengan baik. Output rekomendasi menampilkan buku-buku populer dan beragam genre, yang sesuai dengan minat user berdasarkan buku-buku dengan rating tertinggi yang pernah ia baca. Hal ini menunjukkan bahwa model mampu menangkap pola preferensi user dan menghubungkannya dengan buku-buku lain yang memiliki kemungkinan besar akan disukai, sehingga sistem dapat memberikan rekomendasi yang akurat dan bermanfaat untuk meningkatkan pengalaman membaca pengguna.


