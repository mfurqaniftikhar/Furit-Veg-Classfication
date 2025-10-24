import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


data_cat = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot',
    'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger',
    'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange',
    'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish',
    'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon'
]


st.set_page_config(page_title="Fruit & Vegetable Classifier 🍎", page_icon="🥕", layout="centered")
st.title("🥦 Fruit & Vegetable Classification using CNN")
st.write("Upload an image of a fruit or vegetable and see the model prediction!")


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.keras")  
    return model

model = load_model()


img_height = 180
img_width = 180


uploaded_file = st.file_uploader("📤 Upload a fruit or vegetable image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("🔄 Predicting...")

 
    image = image.resize((img_height, img_width))
    img_arr = tf.keras.utils.img_to_array(image)
    img_bat = tf.expand_dims(img_arr, 0)  


    predcit = model.predict(img_bat)
    score = tf.nn.softmax(predcit[0])

    st.success(f"✅ Predicted: **{data_cat[np.argmax(score)]}**")
    st.info(f"Accuracy: **{np.max(score) * 100:.2f}%**")

    
    st.subheader("Top 3 Predictions:")
    top_3 = np.argsort(score)[-3:][::-1]
    for i in top_3:
        st.write(f"{data_cat[i].capitalize()}: {score[i]*100:.2f}%")


st.markdown("---")
st.caption("🍏 Built with TensorFlow & Streamlit by Furqan Iftikhar")
