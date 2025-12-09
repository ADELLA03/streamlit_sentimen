import pandas as pd
import re
import requests
import nltk
import os # Baru ditambahkan untuk konfigurasi path
from nltk.tokenize import word_tokenize
from io import BytesIO
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ====================================================
# 1. SETUP GLOBAL NLTK, PATH, DAN INISIALISASI OBJEK
# ====================================================

# --- A. Konfigurasi Path NLTK (Penting untuk deployment seperti Streamlit) ---
# Mengatur NLTK_DATA agar resource tersimpan di lokasi yang pasti dapat diakses
if 'NLTK_DATA' not in os.environ:
    os.environ['NLTK_DATA'] = os.path.join(os.getcwd(), 'nltk_data')
    if not os.path.exists(os.environ['NLTK_DATA']):
        os.makedirs(os.environ['NLTK_DATA'])

# --- B. Resource Download (Dilakukan sekali saat script dimulai) ---
# Mengunduh 'punkt'
try:
    nltk.data.find('tokenizers/punkt')
except (nltk.downloader.DownloadError, LookupError):
    print("Mengunduh punkt...")
    nltk.download('punkt')
    
# Mengunduh 'stopwords'
try:
    nltk.data.find('corpora/stopwords')
except (nltk.downloader.DownloadError, LookupError):
    print("Mengunduh stopwords...")
    nltk.download('stopwords')

# --- C. Inisialisasi Objek (Dilakukan sekali setelah download) ---
print("✅ Inisialisasi Stopwords dan Stemmer...")
nltk_stopwords = set(stopwords.words('indonesian')) 
factory = StemmerFactory()
stemmer = factory.create_stemmer()
print("✅ NLTK & Sastrawi siap digunakan.")

# ====================================================
# 2. DEFINISI FUNGSI PREPROCESSING
# ====================================================

## 1. Load dan siapkan data awal
def load_and_prepare(file_path):
    try:
        # Menangani file CSV yang diunggah
        if isinstance(file_path, str) and (file_path.endswith('.csv') or file_path.startswith('http')):
             data = pd.read_csv(file_path, dtype={'created_at': str})
        elif isinstance(file_path, pd.DataFrame):
             data = file_path.copy()
        else:
            raise TypeError("Input harus berupa path file CSV atau DataFrame.")
            
    except Exception as e:
        raise Exception(f"Gagal membaca file CSV: {e}")

    # Konversi tanggal dan jam
    data['created_at'] = pd.to_datetime(data['created_at'], errors='coerce')
    data['tanggal'] = data['created_at'].dt.date
    data['jam'] = data['created_at'].dt.time

    if 'full_text' not in data.columns:
        raise KeyError("Kolom 'full_text' tidak ditemukan di file CSV!")

    df = data[['tanggal', 'jam', 'full_text']].copy()
    df = df.rename(columns={'tanggal': 'date', 'jam': 'time', 'full_text': 'text'})
    return df

## 2. Cleaning text
def clean_text_column(df):
    def remove_URL(text): return re.sub(r'https?://\S+|www\.\S+', '', text)
    def remove_emoji(text):
        emoji_pattern = re.compile("[" 
            u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF" 
            u"\U0001F700-\U0001F77F" u"\U0001F780-\U0001F7FF" u"\U0001F800-\U0001F8FF" 
            u"\U0001F900-\U0001F9FF" u"\U0001FA00-\U0001FA6F" u"\U0001FA70-\U0001FAFF" 
            u"\U0001F004-\U0001F0CF" u"\U0001F1E0-\U0001F1FF" "]+", flags=re.UNICODE)
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

## 3. Case folding
def apply_case_folding(df):
    df['case_folding'] = df['cleaning'].str.lower()
    return df

## 4. Normalisasi kata
def normalize_text(df):
    # Mengambil kamus kata baku dari GitHub
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

## 5. Tokenisasi, stopword removal, stemming + drop duplikat
def tokenize_stop_stem(df):
    # Menggunakan objek 'word_tokenize', 'nltk_stopwords', dan 'stemmer' yang 
    # sudah diinisialisasi di scope global (bagian 1.C)
    
    df['token'] = df['normalisasi'].apply(word_tokenize)
    df['stopword_removal'] = df['token'].apply(lambda x: [w for w in x if w.lower() not in nltk_stopwords])
    df['steming_data'] = df['stopword_removal'].apply(lambda x: ' '.join([stemmer.stem(w) for w in x]))

    # Hapus duplikasi berdasarkan kolom steming_data
    df.drop_duplicates(subset='steming_data', keep='first', inplace=True)

    df['is_unique_stem'] = True
    df.dropna(subset=['steming_data'], inplace=True)
    return df


## 6. Pipeline lengkap preprocessing
def run_full_preprocessing(file_path):
    print("Mulai Preprocessing...")
    
    # Check if NLTK resources were successfully loaded globally
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("ERROR: Resource NLTK 'punkt' GAGAL dimuat. Pastikan koneksi internet stabil saat startup.")
        raise

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
    print(f"Jumlah data akhir: {len(df)}")

    return df

# Contoh penggunaan:
# if __name__ == '__main__':
#     # Ganti 'nama_file_anda.csv' dengan path file Anda
#     # df_final = run_full_preprocessing('nama_file_anda.csv') 
#     pass