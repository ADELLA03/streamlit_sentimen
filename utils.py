import os
import pandas as pd

# Membuat direktori jika belum ada
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# Menyimpan DataFrame ke CSV di folder hasil
def save_csv(df, filename):
    ensure_dir("hasil")
    filepath = os.path.join("hasil", filename)
    df.to_csv(filepath, index=False, encoding='utf8')
    print(f"📄 File CSV disimpan sebagai {filepath}")

# Membaca dataset hasil preprocessing atau labeling (jika sudah ada)
def read_csv_safely(filepath):
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        print(f"⚠️ File {filepath} tidak ditemukan.")
        return None

# Tokenisasi dasar
def tokenize(text):
    return text.split() if isinstance(text, str) else []

# Debug cetak data
def preview_dataframe(df, n=5):
    print("📋 Cuplikan DataFrame:")
    print(df.head(n))
