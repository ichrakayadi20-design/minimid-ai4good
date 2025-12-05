import streamlit as st
from transformers import pipeline
import numpy as np
from PIL import Image
import requests
from io import BytesIO

st.set_page_config(page_title="MiniMind – L'IA expliquée simplement", page_icon="🧠", layout="centered")

st.title("🧠 MiniMind")
st.caption("Par ton équipe – Nuit de l'Info 2025 – Défi AI4GOOD")

st.markdown("""
Salut les jeunes ! Ici tu vas jouer avec l'intelligence artificielle et **voir exactement comment elle pense**.
Trois expériences super simples ↓
""")

# ==================== 1. Chatbot ====================
st.header("1️⃣ Le petit robot qui discute")
st.write("Il s’appelle DialoGPT et il a été entraîné sur des millions de conversations Reddit")

@st.cache_resource
def get_chatbot():
    return pipeline("text-generation", model="microsoft/DialoGPT-small")

chatbot = get_chatbot()

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("Pose-lui une question :", placeholder="Salut, ça va ?")
if question:
    result = chatbot(question, max_length=100, num_return_sequences=1)
    reponse = result[0]['generated_text'].replace(question, "").strip()
    st.session_state.history.append(("Toi", question))
    st.session_state.history.append(("MiniMind", reponse))

for sender, text in st.session_state.history:
    if sender == "Toi":
        st.markdown(f"**🧑 {sender}** : {text}")
    else:
        st.markdown(f"**🤖 {sender}** : {text}")

# ==================== 2. Reconnaissance d’image (sans modèle lourd) ====================
st.header("2️⃣ C’est un chat ou un chien ?")
st.write("J’utilise un modèle très léger fait avec Teachable Machine")

# Modèle pré-entraîné léger hébergé sur Hugging Face (public)
model_url = "https://huggingface.co/spaces/enzostvs/MiniMind-Image-Classifier/resolve/main/keras_model.h5"

@st.cache_resource
def load_model():
    import tensorflow as tf
    return tf.keras.models.load_model("keras_model.h5")  # sera téléchargé automatiquement

model = load_model()
class_names = ["🐱 Chat", "🐶 Chien", "❓ Autre"]

uploaded = st.file_uploader("Envoie une photo", type=["jpg","jpeg","png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB").resize((224,224))
    st.image(image, width=250)
    
    img_array = np.array(image)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred = model.predict(img_array)[0]
    idx = np.argmax(pred)
    st.success(f"Je pense que c’est : **{class_names[idx]}**")
    st.progress(float(pred[idx]))
    
    with st.expander("Comment j’ai décidé ?"):
        for name, proba in zip(class_names, pred):
            st.write(f"{name} : {proba:.1%}")

# ==================== 3. Prédicteur de notes (régression fun) ====================
st.header("3️⃣ Combien d’heures pour avoir 20/20 ?")
st.write("Un petit modèle que j’ai inventé (mais ça marche !)")

col1, col2 = st.columns(2)
with col1:
    etude = st.slider("Heures d’étude par semaine", 0, 40, 15)
with col2:
    sommeil = st.slider("Heures de sommeil par nuit", 4, 12, 8)

note = min(20, etude*0.35 + sommeil*0.8 + 2)
st.metric("Note prédite", f"{note:.1f}/20")

if note >= 16:
    st.balloons()
    st.write("Tu vas tout déchirer !")
elif note < 10:
    st.write("Allez, un petit effort… ou plus de sommeil !")

# ==================== Page explication pédagogique ====================
st.header("Comment ça marche vraiment ?")
st.info("""
• Le chatbot utilise un **Transformer** (comme ChatGPT mais petit)  
• La reconnaissance d’image utilise un **réseau de neurones convolutif (CNN)**  
• La prédiction de note est une **régression linéaire** toute simple  

Tout est open-source et expliqué pour les collégiens !
""")

st.markdown("### Lien GitHub → " + "https://github.com/tonpseudo/minimid-ai4good-2025")
