import streamlit as st
import joblib
import numpy as np

# --- Style CSS ---
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .title {
        color: #0a3d62;
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 0;
    }
    .subtitle {
        color: #3c6382;
        font-size: 20px;
        margin-top: 0;
        margin-bottom: 20px;
    }
    .footer {
        color: #778ca3;
        font-size: 14px;
        margin-top: 30px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    return model

model = load_model()

# Sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/Bank_icon.svg/1024px-Bank_icon.svg.png", width=120)
st.sidebar.title("Crédit Score Predictor")
st.sidebar.markdown("""
Bienvenue !  
Cette application permet de prédire le risque de défaut de crédit  
en se basant sur les caractéristiques client.
""")

# Main title and subtitle
st.markdown('<h1 class="title">Prédiction du défaut de crédit</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Remplissez le formulaire ci-dessous et cliquez sur "Prédire le risque"</p>', unsafe_allow_html=True)

# Illustration
st.image("https://images.unsplash.com/photo-1565372912109-f3d43c0ae2c7?auto=format&fit=crop&w=1050&q=80", use_column_width=True)

# Form container with style
st.markdown("""
<div style="background-color:#ffffff;padding:20px;border-radius:15px;box-shadow:0 4px 8px rgba(0,0,0,0.1);margin-top:20px;">
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    LIMIT_BAL = st.number_input("Montant limite du crédit (LIMIT_BAL)", min_value=1000, max_value=1000000, value=50000, step=1000, help="Montant maximal accordé au client.")
    AGE = st.number_input("Âge du client (AGE)", min_value=18, max_value=100, value=35, step=1, help="Âge en années.")

with col2:
    SEX = st.selectbox("Sexe (SEX)", options=[1, 2], format_func=lambda x: "Homme" if x==1 else "Femme")
    EDUCATION = st.selectbox("Niveau d'éducation (EDUCATION)", options=[1, 2, 3, 4], format_func=lambda x: {1:"Universitaire",2:"Lycée",3:"Collège",4:"Autre"}[x])
    MARRIAGE = st.selectbox("État civil (MARRIAGE)", options=[1, 2, 3], format_func=lambda x: {1:"Marié",2:"Célibataire",3:"Autre"}[x])

st.markdown("</div>", unsafe_allow_html=True)

input_data = np.array([[LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE]])

if st.button("Prédire le risque"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0,1]
    if prediction == 1:
        st.error(f"⚠️ Client à risque de défaut avec une probabilité de {proba:.2f}")
    else:
        st.success(f"✅ Client fiable avec une probabilité de défaut de {proba:.2f}")

st.markdown('<p class="footer">© 2025 Hubert Tchabi - Projet Prédiction Crédit</p>', unsafe_allow_html=True)

