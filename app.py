import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from transformers import pipeline

st.set_page_config(page_title="MiniMind – L'IA expliquée", page_icon="🧠", layout="centered")

st.title("MiniMind – L’IA expliquée simplement")
st.caption("Défi AI4GOOD – Nuit de l’Info 2025 – Équipe [ton nom]")

st.markdown("### 3 expériences pour découvrir l’IA comme un collégien !")

# ===================== 1. CHATBOT (modèle qui marche à 100%) =====================
st.header("1. Le petit robot qui discute")

@st.cache_resource
def get_chatbot():
    # TinyLlama = 1.1B paramètres, ultra-rapide, 100% PyTorch → marche direct sur Streamlit
    return pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype="auto",
        device_map="auto",
        max_new_tokens=100
    )

with st.spinner("Chargement du petit robot… (30-40 sec la première fois)"):
    chatbot = get_chatbot()

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("Pose-lui une question :", placeholder="Salut, tu connais l'IA ?")

if question:
    with st.spinner("MiniMind réfléchit…"):
        result = chatbot(f"<|system|>\nTu es MiniMind, un assistant gentil qui explique l'IA aux enfants.</|system|>\n<|user|>\n{question}</|user|>\n<|assistant|>", 
                         do_sample=True, temperature=0.7)
        reponse = result[0]["generated_text"].split("<|assistant|>")[-1].strip()
    st.session_state.history.append(("Toi", question))
    st.session_state.history.append(("MiniMind", reponse))

for sender, msg in st.session_state.history:
    if sender == "Toi":
        st.markdown(f"**Toi** : {msg}")
    else:
        st.markdown(f"**MiniMind** : {msg}")

# ===================== 2. RECONNAISSANCE D'IMAGES =====================
st.header("2. C’est un chat ou un chien ?")

if "model" not in st.session_state:
    with st.spinner("Chargement du modèle photo…"):
        model = tf.keras.models.load_model("keras_model.h5")
        with open("labels.txt", "r", encoding="utf-8") as f:
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
    st.success(f"C’est un **{class_names[idx]}** !")
    st.progress(float(pred[idx]))
    with st.expander("Détails"):
        for name, p in zip(class_names, pred):
            st.write(f"{name} → {p:.1%}")

# ===================== 3. PRÉDICTEUR DE NOTES =====================
st.header("3. Combien d’heures pour avoir 20/20 ?")
col1, col2 = st.columns(2)
with col1:
    etude = st.slider("Heures d’étude", 0, 50, 20)
with col2:
    sommeil = st.slider("Heures de sommeil", 4, 12, 8)

note = min(20, etude * 0.3 + sommeil * 1.2 + 2)
st.metric("Note prédite", f"{note:.1f}/20")
if note >= 18:
    st.balloons()

# ===================== FIN =====================
st.success("Projet terminé ! Tout fonctionne !")
st.balloons()
