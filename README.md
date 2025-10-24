# 🍎 Fruit & Vegetable Classification using CNN

This project uses a **Convolutional Neural Network (CNN)** built with **TensorFlow / Keras** to classify images of fruits and vegetables into 36 different categories.  
A **Streamlit web app** is included for easy image uploads and real-time predictions.

---

## 🚀 Features

- 🧠 Deep Learning model (CNN) trained on a custom dataset of fruits and vegetables  
- 🎨 User-friendly Streamlit web interface  
- 📸 Upload any fruit/vegetable image for instant prediction  
- 📊 Displays confidence score and top 3 predictions  
- 💾 Deployed locally or on cloud (Streamlit Cloud / Hugging Face / Render)

---

## 🧩 Classes

The model can identify the following categories:
['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot',
'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger',
'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange',
'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish',
'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']

---

## 🧠 Model Overview

The CNN architecture consists of:

- Multiple **Conv2D** and **MaxPooling2D** layers  
- **Flatten** and **Dense** layers for classification  
- **Softmax** activation for final prediction  
- Trained using **categorical cross-entropy loss** and **Adam optimizer**
Streamlit App

The Streamlit app (app.py) allows users to upload an image and view the prediction.

Run the app:
streamlit run app.py

App workflow:

Upload an image (.jpg, .jpeg, .png)

The model predicts the fruit/vegetable class

Displays:

Predicted label

Confidence percentage

Top 3 likely classes

📂 Folder Structure
📁 FruitVeg_Classification/
│
├── 📄 app.py                 # Streamlit web app
├── 📄 model.keras               # Trained CNN model
├── 📄 README.md              # Project documentation
└── 📄 requirements.txt       # Python dependencies


⚙️ Installation
1️⃣ Clone this repository
git clone https://github.com/yourusername/FruitVeg-Classifier.git
cd FruitVeg-Classifier

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the Streamlit app
streamlit run app.py
