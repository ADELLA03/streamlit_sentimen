from sklearn.feature_extraction.text import TfidfVectorizer

def run_tfidf(X_train, X_test):
    """
    Fungsi ini menerima X_train dan X_test sebagai input, dan mengembalikan hasil transformasi 
    dengan menggunakan TF-IDF Vectorizer.
    """
    tfidf_vectorizer = TfidfVectorizer(max_df=0.85, min_df=2, stop_words='english')

    # Fit dan transform data pelatihan, serta transform data uji
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Menyimpan TF-IDF Vectorizer agar bisa digunakan lagi
    return X_train_tfidf, X_test_tfidf, tfidf_vectorizer
