import os, joblib, pandas as pd, streamlit as st
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")
art = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "heart_model.pkl"))
model, enc, FEATURES = art["model"], art["encoders"], art["features"]

st.title("❤️ Heart Disease Predictor")
st.caption(f"Model: {art['model_name']} · uses only 4 inputs")
c1, c2 = st.columns(2)
sex = c1.selectbox("Sex", list(enc["Sex"].classes_))
cp = c2.selectbox("Chest-pain type", list(enc["ChestPain"].classes_))
maxhr = c1.number_input("Max heart rate achieved (bpm)", 60, 220, 150)
oldpeak = c2.number_input("ST depression (Oldpeak)", 0.0, 7.0, 1.0, 0.1)

if st.button("Predict", type="primary"):
    row = pd.DataFrame([{"Sex": enc["Sex"].transform([sex])[0],
                         "ChestPain": enc["ChestPain"].transform([cp])[0],
                         "MaxHR": maxhr, "Oldpeak": oldpeak}])[FEATURES]
    p = model.predict_proba(row)[0, 1]
    (st.error if p >= 0.5 else st.success)(
        f"{'⚠️ Heart disease LIKELY' if p >= 0.5 else '✅ Heart disease UNLIKELY'} — estimated risk {p:.1%}")
st.caption("Educational project – not a medical diagnosis tool.")
