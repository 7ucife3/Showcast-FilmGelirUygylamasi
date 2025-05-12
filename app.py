import streamlit as st
import numpy as np
import pandas as pd
import pickle

# 🌐 Sayfa yapılandırması
st.set_page_config(page_title="ShowCast", layout="centered")

# 🎨 Arka plan ve saydam içerik kutusu CSS
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?fm=jpg&q=60&w=3000&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8bW92aWUlMjBiYWNrZ3JvdW5kfGVufDB8fDB8fHww");
        background-attachment: fixed;
        background-size: cover;
    }
    .main-container {
        background-color: rgba(0, 0, 0, 0.5);
        padding: 2rem;
        border-radius: 12px;
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
    }
    </style>
    <div class="main-container">
    """,
    unsafe_allow_html=True
)

# 🎬 Başlık
st.title("🎬 ShowCast - Film Geliri Tahmini.")

# 📦 Model ve yardımcıları yükle
with open("rf_model_bundle.pkl", "rb") as f:
    saved = pickle.load(f)

rf_model = saved["model"]
le = saved["label_encoder"]
tfidf = saved["tfidf"]
q_low = saved["q_low"]
q_high = saved["q_high"]

# 🔢 Girdi alanları
budget = st.number_input("🎯 Bütçe (USD)", min_value=0, step=1000000, value=50000000)
runtime = st.number_input("🎞️ Süre (dk)", min_value=10, max_value=300, value=120)
vote_avg = st.slider("⭐ Oy Ortalaması", 0.0, 10.0, value=7.0, step=0.1)
vote_count = st.number_input("👥 Oy Sayısı", min_value=0, step=1000, value=10000)
language = st.text_input("🌍 Orijinal Dil (örnek: en)", value="en")
genres = st.text_input("🎭 Türler (örnek: Action Adventure Sci-Fi)", value="Action Adventure")

# 🔮 Tahmin işlemi
if st.button("🎥 Tahmini Geliri Hesapla"):
    try:
        budget_log = np.log1p(budget)
        runtime_winsor = np.clip(runtime, q_low, q_high)
        vote_count_log = np.log1p(vote_count)
        language_encoded = le.transform([language])[0]
        genres_vec = tfidf.transform([genres]).toarray()[0]
        genres_df = pd.DataFrame([genres_vec], columns=[f"genre_{i}" for i in range(len(genres_vec))])

        input_data = pd.DataFrame([[
            budget_log, runtime_winsor, vote_avg, vote_count_log, language_encoded
        ]], columns=['budget_log', 'runtime_winsor', 'vote_average', 'vote_count_log', 'language_encoded'])

        input_full = pd.concat([input_data.reset_index(drop=True), genres_df], axis=1)

        prediction_log = rf_model.predict(input_full)[0]
        prediction = np.expm1(prediction_log)

        st.success(f"💰 Tahmini Gişe Geliri: ${prediction:,.0f}")
        st.write("📊 Log Tahmin:", round(prediction_log, 4))  # Debug için

    except Exception as e:
        st.error(f"Hata oluştu: {e}")

# Saydam kutuyu kapat
st.markdown("</div>", unsafe_allow_html=True)
