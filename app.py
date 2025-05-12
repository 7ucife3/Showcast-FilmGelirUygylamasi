import streamlit as st
import numpy as np
import pickle

# XGBoost modelini yükle (önceden kaydettiysen)
# with open("xgb_model.pkl", "rb") as file:
#     model = pickle.load(file)

# Veya doğrudan eğittiğin modeli içe aktar
from xgboost import XGBRegressor
model = XGBRegressor()
model.load_model("xgb_model.json")  # Modeli JSON olarak kaydettiysen

st.title("🎬 Film Geliri Tahmin Sistemi")

st.subheader("Lütfen film bilgilerini girin:")

budget = st.number_input("Bütçe ($)", min_value=1000, max_value=500_000_000, step=100000)
popularity = st.slider("Popülerlik", min_value=0.0, max_value=800.0, step=1.0)
runtime = st.slider("Süre (dakika)", min_value=30, max_value=300, step=1)
vote_average = st.slider("Oy Ortalaması", min_value=0.0, max_value=10.0, step=0.1)
vote_count = st.number_input("Oy Sayısı", min_value=0, max_value=20000, step=100)

release_year = st.slider("Çıkış Yılı", min_value=1980, max_value=2025, step=1)
release_month = st.selectbox("Ay", list(range(1, 13)))
release_day = st.selectbox("Gün", list(range(1, 32)))

main_genre = st.selectbox("Ana Tür", ['Action', 'Comedy', 'Drama', 'Horror', 'Thriller'])
original_language = st.selectbox("Orijinal Dil", ['en', 'fr', 'es', 'de', 'ja', 'zh'])

if st.button("Tahmini Göster"):
    log_budget = np.log1p(budget)

    input_data = {
        'log_budget': log_budget,
        'popularity': popularity,
        'runtime': runtime,
        'vote_average': vote_average,
        'vote_count': vote_count,
        'release_year': release_year,
        'release_month': release_month,
        'release_day': release_day,
    }

    for lang in ['en', 'fr', 'es', 'de', 'ja', 'zh']:
        input_data[f'original_language_{lang}'] = 1 if original_language == lang else 0

    for genre in ['Action', 'Comedy', 'Drama', 'Horror', 'Thriller']:
        input_data[f'main_genre_{genre}'] = 1 if main_genre == genre else 0

    X_input = np.array([list(input_data.values())])
    log_revenue_pred = model.predict(X_input)[0]
    revenue_pred = np.expm1(log_revenue_pred)

    st.success(f"🎯 Tahmini Gelir: ${revenue_pred:,.0f}")