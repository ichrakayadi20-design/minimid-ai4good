# ==================== 2. Reconnaissance d’image ====================
st.header("2️⃣ C’est un chat ou un chien ?")
st.write("Modèle créé avec Teachable Machine (Google) – entraîné sur vos photos !")

@st.cache_resource
def load_image_model():
    import tensorflow as tf
    model = tf.keras.models.load_model("keras_model.h5")
    with open("labels.txt", "r") as f:
        class_names = [line.strip().split(" ", 1)[1] for line in f.readlines()]
    return model, class_names

model, class_names = load_image_model()

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
