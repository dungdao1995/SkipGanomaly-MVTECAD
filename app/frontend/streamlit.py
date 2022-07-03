import streamlit as st
from PIL import Image
import requests
import time


col1, col2, col3 = st.columns([0.2, 5, 0.2])
img = Image.open('unime.jpeg').resize((128, 128))
col2.image(img,use_column_width=False)
st.title("GAN-based anomaly detection for MVTec Dataset")
st.write('Advisor: **Prof. Dario Bruneo**')
st.write('Co-Advisor: **Fabrizio De Vita**')
st.write('Student: **Dao Khanh Dung - 504416**')

model_name = st.sidebar.selectbox("Select Model", ('GANomaly', 'Skip-Ganomaly'))
st.write("Model: ",model_name)

data_name = st.sidebar.selectbox("Select Data", ('Screw', 'Bottle'))
st.write("Data: ",data_name)

# displays a file uploader widget
image = st.file_uploader("Please upload image", type=["jpg", "png"])

#Display image
if image is not None:
    files = {"file": image.getvalue()}
    img = Image.open(image).resize((256, 256))
    st.image(img, caption='Uploaded Image.', use_column_width=False)
    st.write("Identifying...")
# displays a button
if st.button("Prediction"):
    start_time = time.time()
    res = requests.post(f"http://0.0.0.0:8080/predict", files=files)
    end_time = time.time()
    result = res.json().get('Prediction')
    st.write("Prediction: ", result)
    st.write("Response time: ", end_time -start_time)