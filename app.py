import streamlit as st
from transformers import pipeline
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="MiniMind – L'IA expliquée", page_icon="🧠", layout="centered")

st.title("MiniMind – L’IA expliquée simplement")
st.caption("Défi AI4GOOD – Nuit de l’Info 2025")

st.markdown("### Salut ! Découvre l’IA en jouant avec 3 expériences")

# ===================== 1. CHATBOT =====================
st.header("1. Le petit robot qui discute")
st.write("Modèle : DialoGPT-small (Microsoft)")

@st.cache_resource
def get_chatbot():
    return pipeline("text-generation", model="microsoft/DialoGPT-small")

chatbot = get_chatbot()

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("Pose une question au robot :", placeholder="Salut, ça va ?")

if question:
    result = chatbot(question, max_length=100, num_return_sequences=1)
    reponse = result[0]["generated_text"].replace(question, "").strip()
    st.session_state.history.append(("Toi", question))
    st.session_state.history.append(("MiniMind", reponse))

for sender, msg in st.session_state.history:
    if sender == "Toi":
        st.markdown(f"**Toi** : {msg}")
    else:
        st.markdown(f"**MiniMind** : {msg}")

# ===================== 2. RECONNAISSANCE D'IMAGES =====================
st.header("2. C’est un chat ou un chien ?")

# Chargement du modèle une seule fois
if "model" not in st.session_state:
    with st.spinner("Chargement du modèle d’images… (30 secondes la première fois)"):
        model = tf.keras.models.load_model("keras_model.h5")
        with open("labels.txt", "r", encoding="utf-8") as f:
            # Teachable Machine met parfois "0 Chat", parfois juste "Chat"
            class_names = []
            for line in f:
                parts = line.strip().split(" ", 1)
                class_names.append(parts[1] if len(parts) > 1 else parts[0])
        st.session_state.model = model
        st.session_state.class_names = class_names

model = st.session_state.model
class_names = st.session_state.class_names

uploaded = st.file_uploader("Envoie une photo", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB").resize((224, 224))
    st.image(image, width=300)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0]
    idx = np.argmax(pred)
    confidence = pred[idx]

    st.success(f"Je pense que c’est un **{class_names[idx]}** !")
    st.progress(float(confidence))
    st.write(f"Confiance : {confidence:.1%}")

    with st.expander("Détail des probabilités"):
        for name, proba in zip(class_names, pred):
            st.write(f"{name} → {proba:.1%}")

# ===================== 3. PRÉDICTEUR DE NOTES =====================
st.header("3. Combien d’heures pour avoir 20/20 ?")

col1, col2 = st.columns(2)

with col1:
    etude = st.slider("Heures d’étude par semaine", 0, 50, 20)

with col2:
    sommeil = st.slider("Heures de sommeil par nuit", 4, 12, 8)

note = min(20, etude * 0.3 + sommeil * 1.2 + 2)
st.metric("Note prédite", f"{note:.1f}/20")

if note >= 18:
    st.balloons()
    st.write("Tu vas être major de promo !")
elif note < 8:
    st.write("Attention… plus de sommeil ou plus de révisions !")
else:
    st.write("C’est bien ! Continue comme ça")

# ===================== EXPLICATION PÉDAGOGIQUE =====================
st.header("Comment tout ça marche ?")
st.info("""
• Chatbot → Transformer (comme ChatGPT mais petit)  
• Reconnaissance d’image → Réseau de neurones entraîné avec **Teachable Machine** (Google)  
• Prédicteur de note → Régression linéaire simple  

Tout est gratuit, open-source et pensé pour les collégiens !
""")

st.markdown("### GitHub → https://github.com/[TON_PSEUDO]/minimid-ai4good")
st.markdown("Déployé sur Streamlit Cloud – zéro installation")
st.success("Projet terminé et prêt pour le jury !")
