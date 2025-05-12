import streamlit as st
import numpy as np
from xgboost import XGBRegressor

# Modeli yükle
model = XGBRegressor()
model.load_model("xgb_model.json")

st.title("🎬 Film Geliri Tahmini")

st.markdown("Aşağıdaki film özelliklerini girerek tahmini gelirini öğrenin:")

budget = st.number_input("Bütçe ($)", min_value=1000, step=100000)
runtime = st.number_input("Süre (dakika)", min_value=30, step=1)
vote_average = st.slider("Oy Ortalaması", 0.0, 10.0, step=0.1)
vote_count = st.number_input("Oy Sayısı", min_value=0, step=100)

release_year = st.number_input("Çıkış Yılı", min_value=1980, max_value=2025, step=1)
release_month = st.selectbox("Çıkış Ayı", list(range(1, 13)))
release_day = st.selectbox("Çıkış Günü", list(range(1, 32)))

main_genre = st.selectbox("Ana Tür", ['Action', 'Comedy', 'Drama', 'Horror', 'Thriller'])
original_language = st.selectbox("Orijinal Dil", ['en', 'fr', 'es', 'de', 'ja', 'zh'])

if st.button("Geliri Tahmin Et"):
    log_budget = np.log1p(budget)

    input_data = {
        'log_budget': log_budget,
        'runtime': runtime,
        'vote_average': vote_average,
        'vote_count': vote_count,
        'release_year': release_year,
        'release_month': release_month,
        'release_day': release_day,
    }

    # Kategorik veriler
    for lang in ['en', 'fr', 'es', 'de', 'ja', 'zh']:
        input_data[f'original_language_{lang}'] = 1 if original_language == lang else 0

    for genre in ['Action', 'Comedy', 'Drama', 'Horror', 'Thriller']:
        input_data[f'main_genre_{genre}'] = 1 if main_genre == genre else 0

    # Model tahmini
    X_input = np.array([list(input_data.values())])
    log_revenue_pred = model.predict(X_input)[0]
    revenue_pred = np.expm1(log_revenue_pred)

    st.success(f"Tahmini Gişe Geliri: **${revenue_pred:,.0f}**")
