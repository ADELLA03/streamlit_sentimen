import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords
import os

# --- BAGIAN YANG DIPERBAIKI ---
# 1. Menggunakan stopwords Bahasa Indonesia dari NLTK, sama seperti di preprocessing.py
# Ini akan membuat visualisasi Anda jauh lebih akurat.
try:
    nltk_stopwords = set(stopwords.words('indonesian'))
except LookupError:
    import nltk
    print("Mengunduh stopwords NLTK...")
    nltk.download('stopwords')
    nltk_stopwords = set(stopwords.words('indonesian'))

# Tambahkan kata-kata umum lain yang mungkin ingin Anda abaikan
custom_stopwords = {'yg', 'dgn', 'nya', 'gak', 'ga', 'deh', 'sih', 'aja'}
nltk_stopwords.update(custom_stopwords)
# --- AKHIR BAGIAN PERBAIKAN ---


# Pastikan direktori 'hasil' ada
os.makedirs("hasil", exist_ok=True)

def create_wordcloud(text, filename):
    """Membuat dan menyimpan Word Cloud dari teks."""
    if not text or not text.strip():
        print(f"Tidak ada teks valid untuk membuat Word Cloud {filename}.")
        return

    try:
        # Menggunakan stopwords bahasa Indonesia yang sudah kita siapkan
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            stopwords=nltk_stopwords, # <-- Menggunakan stopwords yang benar
            colormap='viridis',
            max_words=150,
            contour_width=1,
            contour_color='steelblue'
        ).generate(text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        filepath = os.path.join("hasil", filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        print(f"Word Cloud disimpan di {filepath}")
    except Exception as e:
        print(f"Gagal membuat Word Cloud '{filename}': {e}")


def plot_top_words(df, sentiment, filename, n=15):
    """Membuat dan menyimpan plot bar untuk kata-kata yang paling sering muncul."""
    # Memastikan perbandingan sentimen tidak sensitif huruf besar/kecil (lebih aman)
    df['Sentiment'] = df['Sentiment'].str.lower()
    target_df = df[df['Sentiment'] == sentiment.lower()]

    if target_df.empty:
        print(f"Tidak ada data untuk sentimen '{sentiment}' untuk di-plot.")
        return

    # Gabungkan semua teks dan hitung frekuensi kata
    all_words = ' '.join(target_df['steming_data'].astype(str)).split()
    
    # Hapus stopwords bahasa Indonesia
    filtered_words = [word for word in all_words if word not in nltk_stopwords and len(word) > 2]
    
    if not filtered_words:
        print(f"Tidak ada kata tersisa setelah menghapus stopwords untuk sentimen '{sentiment}'.")
        return
        
    word_freq = pd.Series(filtered_words).value_counts().head(n)

    # Buat plot
    plt.figure(figsize=(10, 7))
    sns.barplot(x=word_freq.values, y=word_freq.index, palette='plasma')
    plt.title(f'Top {n} Kata - Sentimen {sentiment.capitalize()}', fontsize=14)
    plt.xlabel('Frekuensi', fontsize=12)
    plt.ylabel('Kata', fontsize=12)
    plt.tight_layout()
    
    filepath = os.path.join("hasil", filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Plot Top Words disimpan di {filepath}")