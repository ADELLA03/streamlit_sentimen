import pandas as pd
import re
import requests
import nltk
from nltk.tokenize import word_tokenize # Sudah ada di sini
from io import BytesIO
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ----------------------------------------------------
# 1. SETUP GLOBAL NLTK DAN INISIALISASI OBJEK
# ----------------------------------------------------

# --- A. Resource Download (Dilakukan sekali saat script dimulai) ---
# Mengunduh 'punkt'
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Mengunduh punkt...")
    nltk.download('punkt')
    
# Mengunduh 'stopwords'
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    print("Mengunduh stopwords...")
    nltk.download('stopwords')

# --- B. Inisialisasi Objek (Dilakukan sekali setelah download) ---
# Inisialisasi Stopwords Indonesia
nltk_stopwords = set(stopwords.words('indonesian'))

# Inisialisasi Stemmer Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# ----------------------------------------------------
# 2. DEFINISI FUNGSI
# ----------------------------------------------------

# 1. Load dan siapkan data awal (Tidak perlu diubah)
def load_and_prepare(file_path):
    try:
        data = pd.read_csv(file_path, dtype={'created_at': str})
    except Exception as e:
        raise Exception(f"Gagal membaca file CSV: {e}")

    # Konversi tanggal dan jam
    data['created_at'] = pd.to_datetime(data['created_at'], errors='coerce')
    data['tanggal'] = data['created_at'].dt.date
    data['jam'] = data['created_at'].dt.time

    # Pastikan kolom full_text ada
    if 'full_text' not in data.columns:
        raise KeyError("Kolom 'full_text' tidak ditemukan di file CSV!")

    # Ambil kolom penting & ubah nama
    df = data[['tanggal', 'jam', 'full_text']].copy()
    df = df.rename(columns={'tanggal': 'date', 'jam': 'time', 'full_text': 'text'})
    return df

# 2. Cleaning text (Tidak perlu diubah)
def clean_text_column(df):
    def remove_URL(text): return re.sub(r'https?://\S+|www\.\S+', '', text)
    def remove_emoji(text):
        emoji_pattern = re.compile("[" 
            u"\U0001F600-\U0001F64F" 
            u"\U0001F300-\U0001F5FF" 
            u"\U0001F680-\U0001F6FF" 
            u"\U0001F700-\U0001F77F" 
            u"\U0001F780-\U0001F7FF" 
            u"\U0001F800-\U0001F8FF" 
            u"\U0001F900-\U0001F9FF" 
            u"\U0001FA00-\U0001FA6F" 
            u"\U0001FA70-\U0001FAFF" 
            u"\U0001F004-\U0001F0CF" 
            u"\U0001F1E0-\U0001F1FF" 
            "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)
    def remove_symbols(text): return re.sub(r'[^a-zA-Z0-9\s]', '', text)
    def remove_numbers(text): return re.sub(r'\d', '', text)
    def deleteHashtag(text): return re.sub(r'#\w+', '', text).strip()
    def deleteUsername(text): return re.sub(r'@\w+', '', text).strip()

    df['cleaning'] = (
        df['text'].astype(str)
        .apply(remove_URL)
        .apply(deleteHashtag)
        .apply(deleteUsername)
        .apply(remove_emoji)
        .apply(remove_symbols)
        .apply(remove_numbers)
    )
    return df

# 3. Case folding (Tidak perlu diubah)
def apply_case_folding(df):
    df['case_folding'] = df['cleaning'].str.lower()
    return df

# 4. Normalisasi kata (Tidak perlu diubah)
def normalize_text(df):
    url = "https://raw.githubusercontent.com/ADELLA03/kamus-normalisasi/main/kamuskatabaku.xlsx"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        kamus_data = pd.read_excel(BytesIO(response.content), engine='openpyxl')
    except Exception as e:
        raise Exception(f"Gagal mengunduh / membaca kamus kata: {e}")

    if not {'tidak_baku', 'kata_baku'}.issubset(kamus_data.columns):
        raise KeyError("Kolom 'tidak_baku' atau 'kata_baku' tidak ditemukan dalam file kamus!")

    kamus_tidak_baku_dict = dict(zip(kamus_data['tidak_baku'], kamus_data['kata_baku']))

    def replace_taboo_words(text):
        words = text.split()
        return ' '.join([kamus_tidak_baku_dict.get(w, w) for w in words])

    df['normalisasi'] = df['case_folding'].apply(replace_taboo_words)
    return df

# 5. Tokenisasi, stopword removal, stemming + drop duplikat (Fungsi yang Diperbaiki)
def tokenize_stop_stem(df):
    # Hapus: from nltk.tokenize import word_tokenize (Sudah di atas)
    # Hapus: nltk.download('stopwords')
    # Hapus: nltk.download('punkt')

    # Hapus: nltk_stopwords = set(stopwords.words('indonesian'))
    # Hapus: factory = StemmerFactory()
    # Hapus: stemmer = factory.create_stemmer()

    # Gunakan objek 'word_tokenize', 'nltk_stopwords', dan 'stemmer' yang sudah
    # diinisialisasi di luar fungsi (di bagian SETUP GLOBAL)
    df['token'] = df['normalisasi'].apply(word_tokenize)
    df['stopword_removal'] = df['token'].apply(lambda x: [w for w in x if w.lower() not in nltk_stopwords])
    df['steming_data'] = df['stopword_removal'].apply(lambda x: ' '.join([stemmer.stem(w) for w in x]))

    # Hapus duplikasi berdasarkan kolom steming_data
    df.drop_duplicates(subset='steming_data', keep='first', inplace=True)

    # Tambahkan kolom indikator bahwa baris ini unik setelah stemming
    df['is_unique_stem'] = True

    df.dropna(subset=['steming_data'], inplace=True)
    return df


# 6. Pipeline lengkap preprocessing (Tidak perlu diubah)
def run_full_preprocessing(file_path):
    df = load_and_prepare(file_path)
    df = clean_text_column(df)
    df = apply_case_folding(df)
    df = normalize_text(df)
    df = tokenize_stop_stem(df)
    df.dropna(inplace=True)

    # Simpan hasil ke file CSV
    output_file = "Hasil_Preprocessing_Data.csv"
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"✅ Preprocessing selesai dan disimpan sebagai {output_file}")

    # Menampilkan hasil akhir
    print(f"Jumlah data akhir: {len(df)}")

    return df