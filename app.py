import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Credit Default Predictor", layout="centered")

st.title("💳 Prédicteur de défaut de crédit")
st.markdown("Prédisez si un client risque de faire défaut à l'aide d'un modèle d'arbre de décision.")

# Chargement du modèle
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# Interface utilisateur (exemple avec 5 features fictives)
st.subheader("📝 Saisissez les informations du client :")

# Générer des champs numériques (remplace par les vrais noms/features si tu les connais)
inputs = []
for i in range(1, 6):
    val = st.number_input(f"Feature {i}", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
    inputs.append(val)

if st.button("Prédire le risque de défaut"):
    prediction = model.predict([np.array(inputs)])
    if prediction[0] == 1:
        st.error("⚠️ Risque de défaut de crédit détecté !")
    else:
        st.success("✅ Client fiable : faible risque de défaut.")
