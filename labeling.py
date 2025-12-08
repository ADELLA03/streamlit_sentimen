import pandas as pd
import requests
from io import BytesIO

# Fungsi untuk memuat lexicon (positif dan negatif) dari URL
# (Kode ini tidak berubah, sudah benar)
def load_lexicon(url):
    lexicon = set()
    try:
        response = requests.get(url)
        df = pd.read_csv(BytesIO(response.content), sep='\t', header=None, names=['word', 'weight'])
        lexicon = set(df['word'])
        print(f"Lexicon loaded from {url}.")
    except Exception as e:
        print(f"Error: Failed to load lexicon from URL: {e}")
    return lexicon

# Memuat lexicon positif dan negatif
# (Kode ini tidak berubah, sudah benar)
positive_url = "https://raw.githubusercontent.com/Taufiq-r/Analisis-sentimen-kaburajadulu/refs/heads/main/kamus/positive.tsv"
negative_url = "https://raw.githubusercontent.com/Taufiq-r/Analisis-sentimen-kaburajadulu/refs/heads/main/kamus/negative.tsv"
positive_words = load_lexicon(positive_url)
negative_words = load_lexicon(negative_url)

# Fungsi untuk memberi label sentimen berdasarkan lexicon
# (Kode ini tidak berubah, sudah benar)
def label_sentiment(tokens, positive_words, negative_words):
    positive_score = sum(1 for token in tokens if token in positive_words)
    negative_score = sum(1 for token in tokens if token in negative_words)
    
    if positive_score > negative_score:
        return 'positif'
    else:
        return 'negatif'

# --- PENYESUAIAN DI SINI ---
# Fungsi sekarang menerima 'df_preprocessed' sebagai input
def run_labeling(df_preprocessed):
    # Kita tidak lagi membaca file, kita gunakan data yang sudah ada di memori
    # Baris 'try...except...pd.read_csv' dihapus
    df = df_preprocessed.copy()

    # Cek apakah kolom 'steming_data' ada
    if 'steming_data' not in df.columns:
        raise ValueError("Column 'steming_data' is missing in the dataframe!")

    # Terapkan pelabelan sentimen (Logika Anda tidak berubah)
    df['Sentiment'] = df['steming_data'].apply(lambda x: label_sentiment(str(x).split(), positive_words, negative_words))

    # Simpan kolom tambahan (Logika Anda tidak berubah)
    df['Score'] = df['Sentiment'].apply(lambda x: 1 if x == 'positif' else 0)

    # --- TAMBAHAN: Simpan file hasil pelabelan ---
    # Ini agar file 'Hasil_Labelling_Data.csv' benar-benar ada
    try:
        df.to_csv("Hasil_Labelling_Data.csv", index=False)
    except Exception as e:
        print(f"Gagal menyimpan Hasil_Labelling_Data.csv: {e}")

    # Pilih kolom yang ingin ditampilkan di output (Logika Anda tidak berubah)
    # Pastikan 'date' dan 'time' ada di df_preprocessed
    # Jika tidak, kita hanya bisa mengembalikan yang ada
    if 'date' in df.columns and 'time' in df.columns:
        result_df = df[['date', 'time', 'steming_data', 'Score', 'Sentiment']]
    else:
        # Fallback jika 'date' atau 'time' tidak ada
        result_df = df[['steming_data', 'Score', 'Sentiment']]

    # Kembalikan DataFrame dengan hasil pelabelan
    return result_df