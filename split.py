# split.py (Ganti semua isinya)

from sklearn.model_selection import train_test_split
import pandas as pd

def split_data(df: pd.DataFrame, test_size: float):
    """
    Membagi DataFrame menjadi data latih dan data uji.

    Args:
        df (pd.DataFrame): DataFrame input.
        test_size (float): Proporsi untuk data uji (contoh: 0.2 untuk 20%).

    Returns:
        tuple: Mengembalikan X_train, X_test, y_train, y_test.
    """
    X = df['steming_data']  # Fitur
    y = df['Sentiment']      # Target/Label
    
    # Memastikan random_state=42 digunakan, sama seperti di notebook
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=42
    )
    
    return X_train, X_test, y_train, y_test