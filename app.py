import streamlit as st
import numpy as np
import pandas as pd
import pickle

# 📦 Kaydedilmiş modeli ve yardımcı nesneleri yükle
with open("rf_model_bundle.pkl", "rb") as f:
    saved = pickle.load(f)

rf_model = saved["model"]
le = saved["label_encoder"]
tfidf = saved["tfidf"]
q_low = saved["q_low"]
q_high = saved["q_high"]




# 🎬 Uygulama başlığı ve açıklama
st.set_page_config(page_title="🎬 ShowCast", layout="centered")
st.title("🎬 ShowCast - Film Geliri Tahmini")
st.markdown("Bu uygulama, verdiğiniz film bilgilerine göre tahmini gişe gelirini hesaplar. Random Forest algoritması kullanılmıştır.")

# 🔢 Kullanıcı giriş alanları
budget = st.number_input("🎯 Bütçe (USD)", min_value=0, step=1000000, value=50000000)
runtime = st.number_input("🎞️ Süre (dk)", min_value=10, max_value=300, value=120)
vote_avg = st.slider("⭐ Oy Ortalaması", 0.0, 10.0, value=7.0, step=0.1)
vote_count = st.number_input("👥 Oy Sayısı", min_value=0, step=1000, value=10000)
language = st.text_input("🌍 Orijinal Dil (örnek: en)", value="en")
genres = st.text_input("🎭 Türler (örnek: Action Adventure Sci-Fi)", value="Action Adventure")

# 📌 Tahmin butonu
if st.button("🎥 Tahmini Geliri Hesapla"):
    try:
        # Girdi dönüşümleri
        budget_log = np.log1p(budget)
        runtime_winsor = np.clip(runtime, q_low, q_high)
        vote_count_log = np.log1p(vote_count)
        language_encoded = le.transform([language])[0]
        genres_vec = tfidf.transform([genres]).toarray()[0]
        genres_df = pd.DataFrame([genres_vec], columns=[f"genre_{i}" for i in range(len(genres_vec))])

        # Final girdi vektörü
        input_data = pd.DataFrame([[
            budget_log, runtime_winsor, vote_avg, vote_count_log, language_encoded
        ]], columns=['budget_log', 'runtime_winsor', 'vote_average', 'vote_count_log', 'language_encoded'])

        input_full = pd.concat([input_data.reset_index(drop=True), genres_df], axis=1)

        # 🔮 Tahmin
        prediction_log = rf_model.predict(input_full)[0]
        prediction = np.expm1(prediction_log)

        st.success(f"💰 Tahmini Gişe Geliri: ${prediction:,.0f}")
    except Exception as e:
        st.error(f"Hata oluştu: {e}")
