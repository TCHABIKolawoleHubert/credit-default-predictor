import streamlit as st
import joblib
import numpy as np

# Configuration de la page
st.set_page_config(
    page_title="Prédicteur de défaut de crédit",
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS pour un design plus moderne
st.markdown("""
    <style>
        body {
            background-color: #f4f6f9;
        }
        .stApp {
            background-image: linear-gradient(to right top, #dfe9f3, #ffffff);
            padding: 2rem;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Titre et description
st.title("💳 Prédicteur de défaut de crédit")
st.markdown("Utilisez ce modèle d'arbre de décision pour estimer le **risque de défaut** d'un client bancaire.")

st.markdown("---")

# Fonction de chargement du modèle
@st.cache_resource
def load_model():
    try:
        return joblib.load("model.pkl")
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

model = load_model()

# Saisie des caractéristiques du client
st.subheader("📋 Informations client")

# Remplace ces noms par les vrais si tu les connais
feature_names = [
    "Montant du crédit",
    "Durée de remboursement (mois)",
    "Âge du client",
    "Nombre de crédits en cours",
    "Revenus mensuels"
]

inputs = []
for name in feature_names:
    val = st.number_input(name, min_value=0.0, max_value=100000.0, value=1000.0, step=100.0)
    inputs.append(val)

# Bouton de prédiction
if st.button("📊 Prédire le risque de défaut"):
    if model is not None:
        try:
            input_array = np.array(inputs).reshape(1, -1)
            prediction = model.predict(input_array)
            if prediction[0] == 1:
                st.error("❌ Risque élevé de **défaut de crédit** détecté !")
            else:
                st.success("✅ Client considéré comme **fiable** : faible risque de défaut.")
        except Exception as e:
            st.warning(f"Erreur lors de la prédiction : {e}")
    else:
        st.warning("Le modèle n'a pas pu être chargé.")
