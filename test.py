from ultralytics import YOLO
import cv2
import streamlit as st 
from PIL import Image
import numpy as np

# imgpath = r"C:\Users\Admin\Desktop\test data\catt.jpg"
modelpath = "best.pt"
# img = cv2.imread(imgpath)
model = YOLO(modelpath)

st.title('Insert your image for prediction (dog or cat)')
image = st.file_uploader('upload image',type=['png', 'jpg', 'jpeg', 'gif'])
if image:
    image = Image.open(image)
    st.image(image=image)
    result = model(image)
    names = result[0].names
    probability = result[0].probs.data.numpy()
    prediction = np.argmax(probability)
    st.write(names)
    st.write(prediction)
    if prediction == 0:
        st.write('This is a CAT')
    else:st.write('This is a DOG')
    




# print(result)
