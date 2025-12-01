import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Deteksi Tingkat Depresi Siswa", layout="wide")

st.title("🧠 Deteksi Tingkat Depresi Siswa (C4.5 & Random Forest)")

uploaded_file = st.file_uploader("📂 Upload file CSV", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if data.shape[0] < 5 or data.shape[1] < 2:
        st.error("❗ Data terlalu sedikit atau format tidak sesuai. Pastikan minimal 5 baris dan ada label target.")
        st.stop()

    st.subheader("📄 Data Awal")
    st.dataframe(data)

    # Encoding
    data_encoded = data.copy()
    label_encoders = {}
    for col in data_encoded.columns:
        if data_encoded[col].dtype == 'object':
            le = LabelEncoder()
            data_encoded[col] = le.fit_transform(data_encoded[col].astype(str))
            label_encoders[col] = le

    st.subheader("🔤 Data Setelah Encoding")
    st.dataframe(data_encoded)

    X = data_encoded.iloc[:, :-1]
    y = data_encoded.iloc[:, -1]

    # Validasi sebelum modeling
    if X.isnull().any().any() or y.isnull().any():
        st.error("❗ Data mengandung nilai kosong. Harap bersihkan dataset.")
        st.stop()

    if len(np.unique(y)) < 2:
        st.error("❗ Label target hanya memiliki 1 kelas. Model tidak dapat dilatih.")
        st.stop()

    # Model setup
    dt_model = DecisionTreeClassifier(criterion="entropy", random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Cross Val Score
    dt_scores = cross_val_score(dt_model, X, y, cv=kf, error_score='raise')
    rf_scores = cross_val_score(rf_model, X, y, cv=kf, error_score='raise')

    st.subheader("📊 Akurasi Tiap Fold - Decision Tree (C4.5)")
    st.dataframe(pd.DataFrame({'Fold': range(1, 6), 'Akurasi': dt_scores}))
    st.write(f"**Rata-rata Akurasi Decision Tree:** {np.mean(dt_scores):.2f}")

    st.subheader("📊 Akurasi Tiap Fold - Random Forest")
    st.dataframe(pd.DataFrame({'Fold': range(1, 6), 'Akurasi': rf_scores}))
    st.write(f"**Rata-rata Akurasi Random Forest:** {np.mean(rf_scores):.2f}")

    # Confusion Matrix
    y_pred_dt = cross_val_predict(dt_model, X, y, cv=kf)
    y_pred_rf = cross_val_predict(rf_model, X, y, cv=kf)

    st.subheader("🟦 Confusion Matrix - Decision Tree")
    fig_dt, ax_dt = plt.subplots()
    sns.heatmap(confusion_matrix(y, y_pred_dt), annot=True, fmt='d', cmap='Blues', ax=ax_dt)
    st.pyplot(fig_dt)

    st.subheader("🟩 Confusion Matrix - Random Forest")
    fig_rf, ax_rf = plt.subplots()
    sns.heatmap(confusion_matrix(y, y_pred_rf), annot=True, fmt='d', cmap='Greens', ax=ax_rf)
    st.pyplot(fig_rf)

    # Classification Report
    st.subheader("📝 Classification Report - Decision Tree (C4.5)")
    st.text(classification_report(y, y_pred_dt))

    st.subheader("📝 Classification Report - Random Forest")
    st.text(classification_report(y, y_pred_rf))

    # Data hasil prediksi
    st.subheader("📋 Data dengan Label Asli & Prediksi")
    hasil_df = X.copy()
    hasil_df['Label_Asli'] = y
    hasil_df['Prediksi_DT'] = y_pred_dt
    hasil_df['Prediksi_RF'] = y_pred_rf
    st.dataframe(hasil_df)

    # Visualisasi Tree
    st.subheader("🌳 Visualisasi Pohon Keputusan")
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    dt_model.fit(X_train, y_train)
    fig_tree, ax_tree = plt.subplots(figsize=(12, 8))
    plot_tree(dt_model, feature_names=X.columns, class_names=[str(c) for c in np.unique(y)], filled=True, ax=ax_tree)
    st.pyplot(fig_tree)

    # Prediksi Data Baru
    st.subheader("📝 Prediksi Data Baru")
    input_values = []
    cols_input = st.columns(len(X.columns))
    for idx, col in enumerate(X.columns):
        val = cols_input[idx].number_input(f"{col}", value=0.0)
        input_values.append(val)

    if st.button("🔮 Prediksi"):
        new_data = pd.DataFrame([input_values], columns=X.columns)
        pred_dt = dt_model.predict(new_data)[0]
        rf_model.fit(X, y)
        pred_rf = rf_model.predict(new_data)[0]
        st.success(f"📌 Prediksi Decision Tree (C4.5): **{pred_dt}**")
        st.success(f"📌 Prediksi Random Forest: **{pred_rf}**")

else:
    st.info("🗂️ Silakan upload file CSV terlebih dahulu.")