import streamlit as st
from io import BytesIO
from PIL import Image
import numpy as np
import torch
from torchvision import transforms


if 'model' not in st.session_state:
    st.session_state['model'] = torch.load('models/model.pth',map_location=torch.device('cpu'))

transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


st.write('Our Model is finetuned VGG13')


st.write("Please upload picture of your animal here:")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    st.image(bytes_data)
    st.session_state['image'] = Image.open(BytesIO(bytes_data))


if 'image' in st.session_state: 
    if st.button('Predict the animal name'):
        classes = ['cheetah', 'fox', 'hyena', 'lion', 'tiger', 'wolf']
        transformed_image = transform(st.session_state['image'])
        logits = st.session_state['model'](transformed_image [None,...])
        precited_class = classes[logits.argmax().item()]
        confidence = torch.nn.functional.softmax(logits).max().item()*100
        st.write(f"You gave image of a {precited_class} , our model is {confidence:.2f}% sure")
