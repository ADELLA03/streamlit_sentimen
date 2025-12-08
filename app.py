import os
import nltk
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import re
from split import split_data
from preprocessing import run_full_preprocessing
from labeling import run_labeling
from modeling import run_naive_bayes
from modeling_knn import run_knn
from modeling_svm import run_svm
# --- TAMBAHAN BARU ---
from visualization import create_wordcloud, plot_top_words
from sklearn.feature_extraction.text import TfidfVectorizer

# ✅ Konfigurasi Streamlit & Logo
st.set_page_config(page_title="Analisis Sentimen", page_icon="✨", layout="wide")

# --- Pengaturan Awal ---
os.makedirs("hasil", exist_ok=True)

# Unduh stopwords bila belum ada
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    try:
        nltk.download("stopwords", quiet=True)
    except Exception as e:
        st.warning(f"Gagal mengunduh stopwords NLTK: {e}")

# --- Fungsi Helper ---
def parse_classification_report(report):
    lines = report.split('\n')
    data = []
    target_keywords = ['negatif', 'positif', 'macro avg', 'weighted avg']
    
    for line in lines:
        line_stripped = line.strip()
        if any(line_stripped.startswith(keyword) for keyword in target_keywords):
            parts = re.split(r'\s{2,}', line_stripped)
            if len(parts) >= 5:
                label = parts[0]
                precision = float(parts[1])
                recall = float(parts[2])
                f1_score = float(parts[3])
                support = int(parts[4])
                
                if "macro avg" in label: label = "Macro Avg"
                elif "weighted avg" in label: label = "Weighted Avg"
                else: label = label.capitalize()
                
                data.append([label, precision, recall, f1_score, support])
            
    return pd.DataFrame(data, columns=['Sentiment', 'Precision', 'Recall', 'F1-Score', 'Support'])

# ===========================
# Judul & Tab
# ===========================
st.title("📊 Analisis Sentimen (Naive Bayes, SVM & KNN)")

# --- PENAMBAHAN 'visual_tab' ---
upload_tab, preprocess_tab, label_tab, model_tab, visual_tab = st.tabs([
    "📂 Upload Data",
    "🔄 Preprocessing",
    "🏷️ Labeling & Split",
    "📈 Modeling (NB, SVM & KNN)",
    "🖼️ Visualisasi"
])

# ===========================
# TAB 1: UPLOAD DATA
# ===========================
with upload_tab:
    st.header("📂 Unggah File CSV Anda")
    uploaded_file = st.file_uploader("Pilih file CSV yang berisi data tweet", type="csv", key="uploader_csv")
    if uploaded_file:
        with open("preprocessing.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("✅ File berhasil diunggah dan disimpan sebagai `preprocessing.csv`. Silakan lanjut ke tab Preprocessing.")

# ===========================
# TAB 2: PREPROCESSING
# ===========================
with preprocess_tab:
    st.header("🔄 Tahap Preprocessing Data Teks")
    is_disabled = not os.path.exists("preprocessing.csv")
    if is_disabled:
        st.info("Silakan unggah file CSV di tab 'Upload Data' terlebih dahulu.")
    
    if st.button("🚀 Jalankan Semua Proses Preprocessing", key="btn_preprocess", disabled=is_disabled):
        with st.spinner("Sedang membersihkan data, mohon tunggu..."):
            try:
                df_preprocessed = run_full_preprocessing("preprocessing.csv")
                st.session_state.df_preprocessed = df_preprocessed

                # Simpan hasil preprocessing ke file CSV agar bisa dilihat/diunduh.
                df_preprocessed.to_csv("Hasil_Preprocessing_Data.csv", index=False)
                
                st.success("✅ Preprocessing selesai. Hasil disimpan di `Hasil_Preprocessing_Data.csv`.")
            except Exception as e:
                st.error(f"Gagal melakukan preprocessing: {e}")

    if 'df_preprocessed' in st.session_state:
        with st.expander("📄 Lihat Hasil Preprocessing (5 baris pertama)") :
            st.dataframe(st.session_state.df_preprocessed.head())
        if os.path.exists("Hasil_Preprocessing_Data.csv"):
            with open("Hasil_Preprocessing_Data.csv", "rb") as file:
                st.download_button(label="📥 Download Hasil Preprocessing", data=file, file_name="Hasil_Preprocessing_Data.csv", mime="text/csv", key="download_after_preprocessing")

# ===========================
# TAB 3: LABELING & SPLIT
# ===========================
with label_tab:
    st.header("🏷️ Tahap Pelabelan Sentimen (Lexicon-Based)")
    # Cek apakah hasil preprocessing sudah ada di session_state
    labeling_disabled = 'df_preprocessed' not in st.session_state

    if labeling_disabled:
        st.info("Jalankan proses preprocessing terlebih dahulu di tab sebelumnya.")

    if st.button("🏷️ Jalankan Pelabelan Sentimen", key="btn_label", disabled=labeling_disabled):
        with st.spinner("Memberi label sentimen pada setiap data..."):
            try:
                df_labelled = run_labeling(st.session_state.df_preprocessed)
                st.session_state.df_labelled = df_labelled
                
                # --- TAMBAHAN BARU: Simpan hasil pelabelan untuk tab visualisasi ---
                df_labelled.to_csv("Hasil_Labelling_Data.csv", index=False)
                
                st.success("✅ Pelabelan selesai. Hasil disimpan di `Hasil_Labelling_Data.csv`.")
            except Exception as e:
                st.error(f"Gagal melakukan pelabelan: {e}")

    if 'df_labelled' in st.session_state:
        with st.expander("📄 Lihat Hasil Pelabelan (5 baris pertama)") :
            display_cols = [col for col in ['date', 'time', 'steming_data', 'Score', 'Sentiment'] if col in st.session_state.df_labelled.columns]
            st.dataframe(st.session_state.df_labelled[display_cols].head())

        st.write(f"Jumlah Total Data Setelah Dilabeli: **{len(st.session_state.df_labelled)}**")

        sentiment_distribution = st.session_state.df_labelled['Sentiment'].value_counts()
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = ["green" if lbl == "positif" else "red" for lbl in sentiment_distribution.index]
        bars = ax.bar(sentiment_distribution.index, sentiment_distribution.values, color=colors)
        ax.set_title('Distribusi Sentimen Hasil Pelabelan Lexicon')
        ax.set_xlabel('Sentimen'); ax.set_ylabel('Jumlah Tweet')
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 5, int(yval), ha='center', va='bottom')
        st.pyplot(fig)

        st.markdown("---")
        st.header("🔀 Pembagian Data (Train & Test)")

        split_options = { "60% Latih : 40% Uji": 0.4, "70% Latih : 30% Uji": 0.3, "80% Latih : 20% Uji": 0.2, "90% Latih : 10% Uji": 0.1 }
        split_choice = st.selectbox("Pilih rasio pembagian data", list(split_options.keys()), key="split_ratio", index=2)

        if st.button("🔀 Bagi Data", key="btn_split"):
            correct_test_size = split_options[split_choice]
            X_train, X_test, y_train, y_test = split_data(st.session_state.df_labelled, correct_test_size)
            train_df = pd.DataFrame({'steming_data': X_train, 'Sentiment': y_train})
            test_df = pd.DataFrame({'steming_data': X_test, 'Sentiment': y_test})
            st.session_state.train = train_df
            st.session_state.test = test_df

            with st.spinner("Membuat 'Master' TF-IDF Vectorizer berdasarkan data 90%..."):
                X_train_90, _, _, _ = split_data(st.session_state.df_labelled, 0.1)
                master_vectorizer = TfidfVectorizer()
                master_vectorizer.fit(X_train_90)

            st.session_state.master_tfidf_vectorizer = master_vectorizer
            st.success("✅ 'Master' TF-IDF Vectorizer berhasil dibuat dan disimpan.")

            st.success(f"Data berhasil dibagi → Data Latih: {len(train_df)}, Data Uji: {len(test_df)}")
            st.write("Data Latih (5 baris pertama):"); st.dataframe(train_df.head())
            st.write("Data Uji (5 baris pertama):"); st.dataframe(test_df.head())

# ===========================
# TAB 4: MODELING
# ===========================
with model_tab:
    st.header("📈 Pemodelan dan Evaluasi Klasifikasi")
    st.subheader("⚙️ Opsi Penyeimbangan Data")

    use_smote_option = st.checkbox("Terapkan SMOTE pada data training", key="smote_checkbox", help="SMOTE akan menyeimbangkan jumlah data pada kelas minoritas (oversampling).")
    st.markdown("---")
    
    model_disabled = not all(k in st.session_state for k in ['train', 'test', 'master_tfidf_vectorizer'])
    if model_disabled:
        st.warning("Silakan jalankan proses 'Pelabelan & Split' terlebih dahulu untuk membuat data dan 'Master' TF-IDF.")

    # === Naive Bayes
    st.subheader("1. Naive Bayes (Multinomial)")
    if st.button("🔍 Latih & Uji Model Naive Bayes", key="btn_nb", disabled=model_disabled):
        with st.spinner("Sedang melatih dan mengevaluasi model Naive Bayes..."):
            try:
                (accuracy, report, train_n, test_n, vocab_size, n_classes, conf_path, vecinfo, smote_info, report_path_nb, pred_path_nb, model_path_nb, vec_path_nb) = run_naive_bayes(
                    st.session_state.train, st.session_state.test, st.session_state.master_tfidf_vectorizer, use_smote=use_smote_option
                )
                st.session_state.nb_accuracy = accuracy
                st.session_state.nb_report = report
                st.session_state.nb_smote_info = smote_info
                st.session_state.nb_paths = {"conf_matrix": conf_path, "report": report_path_nb, "pred": pred_path_nb, "model": model_path_nb, "vec": vec_path_nb}
                st.success("✅ Pelatihan & Pengujian Naive Bayes Selesai!")
            except Exception as e: st.error(f"Error in Naive Bayes: {e}")

    if 'nb_report' in st.session_state:
        st.markdown("##### Hasil Evaluasi Naive Bayes")
        if st.session_state.get('nb_smote_info'):
            st.info(st.session_state.nb_smote_info)
        col_acc, col_rep = st.columns([1, 3])
        with col_acc: st.metric("Akurasi", f"{st.session_state.nb_accuracy:.4f}")
        with col_rep: st.dataframe(parse_classification_report(st.session_state.nb_report), hide_index=True)
        st.image(st.session_state.nb_paths["conf_matrix"], caption="Confusion Matrix Naive Bayes")

    # === SVM (PERUBAHAN URUTAN 2) ===
    st.subheader("2. SVM (Support Vector Machine)")
    if st.button("🔍 Latih & Uji Model SVM", key="btn_svm", disabled=model_disabled):
        with st.spinner("Sedang melatih dan mengevaluasi model SVM..."):
            try:
                (accuracy, report, train_n, test_n, vocab_size, n_classes, conf_path, vecinfo, smote_info, report_path_svm, pred_path_svm, model_path_svm, vec_path_svm) = run_svm(
                    st.session_state.train, st.session_state.test, st.session_state.master_tfidf_vectorizer, use_smote=use_smote_option
                )
                st.session_state.svm_accuracy = accuracy
                st.session_state.svm_report = report
                st.session_state.svm_smote_info = smote_info
                st.session_state.svm_paths = {"conf_matrix": conf_path, "report": report_path_svm, "pred": pred_path_svm, "model": model_path_svm, "vec": vec_path_svm}
                st.success("✅ Pelatihan & Pengujian SVM Selesai!")
            except Exception as e: st.error(f"Error in SVM: {e}")

    if 'svm_report' in st.session_state:
        st.markdown("##### Hasil Evaluasi SVM")
        if st.session_state.get('svm_smote_info'):
            st.info(st.session_state.svm_smote_info)
        col_acc, col_rep = st.columns([1, 3])
        with col_acc: st.metric("Akurasi", f"{st.session_state.svm_accuracy:.4f}")
        with col_rep: st.dataframe(parse_classification_report(st.session_state.svm_report), hide_index=True)
        st.image(st.session_state.svm_paths["conf_matrix"], caption="Confusion Matrix SVM")

    # === KNN (PERUBAHAN URUTAN 3) ===
    st.subheader("3. KNN (K-Nearest Neighbors)")
    if st.button("🔍 Latih & Uji Model KNN", key="btn_knn", disabled=model_disabled):
        with st.spinner("Sedang melatih dan mengevaluasi model KNN..."):
            try:
                (accuracy, report, train_n, test_n, vocab_size, n_classes, conf_path, vecinfo, smote_info, report_path_knn, pred_path_knn, model_path_knn, vec_path_knn) = run_knn(
                    st.session_state.train, st.session_state.test, st.session_state.master_tfidf_vectorizer, use_smote=use_smote_option
                )
                st.session_state.knn_accuracy = accuracy
                st.session_state.knn_report = report
                st.session_state.knn_smote_info = smote_info
                st.session_state.knn_paths = {"conf_matrix": conf_path, "report": report_path_knn, "pred": pred_path_knn, "model": model_path_knn, "vec": vec_path_knn}
                st.success("✅ Pelatihan & Pengujian KNN Selesai!")
            except Exception as e: st.error(f"Error in KNN: {e}")

    if 'knn_report' in st.session_state:
        st.markdown("##### Hasil Evaluasi KNN")
        if st.session_state.get('knn_smote_info'):
            st.info(st.session_state.knn_smote_info)
        col_acc, col_rep = st.columns([1, 3])
        with col_acc: st.metric("Akurasi", f"{st.session_state.knn_accuracy:.4f}")
        with col_rep: st.dataframe(parse_classification_report(st.session_state.knn_report), hide_index=True)
        st.image(st.session_state.knn_paths["conf_matrix"], caption="Confusion Matrix KNN")

    st.markdown("---")

# ===========================
# TAB 5: VISUALISASI (BLOK BARU)
# ===========================
with visual_tab:
    st.header("🖼️ Visualisasi Hasil Analisis")
    
    visual_disabled = not os.path.exists("Hasil_Labelling_Data.csv")
    if visual_disabled:
        st.warning("Jalankan 'Pelabelan Sentimen' di tab sebelumnya untuk mengaktifkan visualisasi.")
    else:
        # Pindahkan tombol ke sini agar tidak tersembunyi oleh warning
        if st.button("📊 Buat & Tampilkan Visualisasi", key="btn_viz", disabled=visual_disabled):
            with st.spinner("Membuat Word Cloud dan Grafik Top Words..."):
                try:
                    df_vis = pd.read_csv("Hasil_Labelling_Data.csv")
                    if 'steming_data' not in df_vis.columns or 'Sentiment' not in df_vis.columns:
                        st.error("File tidak memiliki kolom 'steming_data' atau 'Sentiment'.")
                    else:
                        # Pastikan kolom 'Sentiment' dalam huruf kecil untuk konsistensi
                        df_vis['Sentiment'] = df_vis['Sentiment'].str.lower()
                        
                        positif_text = ' '.join(df_vis[df_vis['Sentiment'] == 'positif']['steming_data'].astype(str))
                        negatif_text = ' '.join(df_vis[df_vis['Sentiment'] == 'negatif']['steming_data'].astype(str))
                        
                        # Buat visualisasi hanya jika ada teks
                        if positif_text.strip():
                            create_wordcloud(positif_text, "wordcloud_positif.png")
                            plot_top_words(df_vis, 'positif', "top_words_positif.png")
                        
                        if negatif_text.strip():
                            create_wordcloud(negatif_text, "wordcloud_negatif.png")
                            plot_top_words(df_vis, 'negatif', "top_words_negatif.png")
                        
                        st.session_state.visuals_generated = True
                        st.success("✅ Visualisasi berhasil dibuat!")
                except FileNotFoundError:
                    st.error("File `Hasil_Labelling_Data.csv` tidak ditemukan.")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat membuat visualisasi: {e}")

    if st.session_state.get('visuals_generated'):
        st.subheader("☁️ Word Cloud")
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists("hasil/wordcloud_positif.png"):
                st.image("hasil/wordcloud_positif.png", caption="Word Cloud Positif")
            else:
                st.info("Tidak ada data positif untuk membuat Word Cloud.")
        with col2:
            if os.path.exists("hasil/wordcloud_negatif.png"):
                st.image("hasil/wordcloud_negatif.png", caption="Word Cloud Negatif")
            else:
                st.info("Tidak ada data negatif untuk membuat Word Cloud.")
        
        st.markdown("---")
        st.subheader("📊 Top 15 Kata yang Paling Sering Muncul")
        col3, col4 = st.columns(2)
        with col3:
            if os.path.exists("hasil/top_words_positif.png"):
                st.image("hasil/top_words_positif.png", caption="Top Words Positif")
            else:
                st.info("Tidak ada data positif untuk membuat grafik.")
        with col4:
            if os.path.exists("hasil/top_words_negatif.png"):
                st.image("hasil/top_words_negatif.png", caption="Top Words Negatif")
            else:
                st.info("Tidak ada data negatif untuk membuat grafik.")
                
    st.markdown("---")
    
    st.header("⚖️ Perbandingan Performa Model")
    # === PERUBAHAN URUTAN DI DICTIONARY ===
    accuracies = {
        'Naive Bayes': st.session_state.get('nb_accuracy'),
        'SVM': st.session_state.get('svm_accuracy'),
        'KNN': st.session_state.get('knn_accuracy')
    }
    
    valid_accuracies = {model: acc for model, acc in accuracies.items() if acc is not None}
    
    if not valid_accuracies:
        st.info("Jalankan satu atau lebih model di tab 'Modeling' untuk melihat perbandingan performa.")
    else:
        df_accuracy = pd.DataFrame(list(valid_accuracies.items()), columns=['Model', 'Akurasi']).sort_values('Akurasi', ascending=False)
        # === PERUBAHAN URUTAN DI DICTIONARY ===
        color_map = {'Naive Bayes': '#ff9999', 'SVM': '#99ff99', 'KNN': '#66b3ff'}
        bar_colors = [color_map.get(model, '#cccccc') for model in df_accuracy['Model']]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(df_accuracy['Model'], df_accuracy['Akurasi'], color=bar_colors)
        ax.set_title('Perbandingan Akurasi Antar Model')
        ax.set_ylabel('Akurasi')
        ax.set_ylim(0, 1.05)
        
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom')
            
        st.pyplot(fig)