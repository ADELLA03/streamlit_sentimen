from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import pandas as pd
from smote import apply_smote

RESULT_DIR = 'hasil'

def run_knn(train, test, vectorizer, use_smote=False, n_neighbors=5):
    os.makedirs(RESULT_DIR, exist_ok=True)

    X_train = train["steming_data"].astype(str)
    y_train = train["Sentiment"].astype(str)
    X_test = test["steming_data"].astype(str)
    y_test = test["Sentiment"].astype(str)

    X_tfidf_train = vectorizer.transform(X_train)
    X_tfidf_test = vectorizer.transform(X_test)
    
    # +++ TAMBAHAN: Variabel untuk menyimpan info SMOTE +++
    smote_info = None
    
    if use_smote:
        counts_before = y_train.value_counts()
        X_tfidf_train, y_train = apply_smote(X_tfidf_train, y_train)
        counts_after = pd.Series(y_train).value_counts()
        smote_info = (
            f"Data Latih Sebelum SMOTE: Positif={counts_before.get('positif', 0)}, Negatif={counts_before.get('negatif', 0)}. "
            f"Setelah SMOTE: Positif={counts_after.get('positif', 0)}, Negatif={counts_after.get('negatif', 0)}."
        )

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_tfidf_train, y_train)
    y_pred = knn.predict(X_tfidf_test)
    accuracy = knn.score(X_tfidf_test, y_test)
    report = classification_report(y_test, y_pred, target_names=sorted(y_train.unique()), zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred, labels=sorted(y_train.unique()))
    conf_path = os.path.join(RESULT_DIR, "conf_matrix_knn.png")
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=sorted(y_train.unique()), yticklabels=sorted(y_train.unique()))
    plt.title("Confusion Matrix - KNN")
    plt.xlabel("Prediksi"); plt.ylabel("Aktual")
    plt.tight_layout(); plt.savefig(conf_path); plt.close()

    report_path = os.path.join(RESULT_DIR, "report_knn.txt")
    with open(report_path, "w", encoding="utf-8") as f: f.write(report)
    pred_path = os.path.join(RESULT_DIR, "pred_knn.csv")
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(pred_path, index=False)
    model_path = os.path.join(RESULT_DIR, "model_knn.joblib")
    vec_path = os.path.join(RESULT_DIR, "tfidf_vectorizer.joblib")
    joblib.dump(knn, model_path)
    joblib.dump(vectorizer, vec_path)
    
    train_n = X_tfidf_train.shape[0]
    test_n = X_tfidf_test.shape[0]
    vocab_size = len(vectorizer.vocabulary_)
    n_classes = len(knn.classes_)
    vecinfo = f"TF-IDF (Vocab Size: {vocab_size})"

    # --- PENYESUAIAN: Kembalikan 13 nilai (termasuk smote_info) ---
    return (accuracy, report, train_n, test_n, vocab_size, n_classes,
            conf_path, vecinfo, smote_info, report_path, pred_path, model_path, vec_path)