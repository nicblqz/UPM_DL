import streamlit as st
import numpy as np
import torch
import cv2
from PIL import Image
import os
import io
from torchvision import models, transforms
import torch.nn as nn
from torchvision.io import read_image, ImageReadMode
from skimage.color import lab2rgb
from streamlit_image_comparison import image_comparison

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 131)  # Update for number of classes (Fruits-360 has 131 classes)
    model.load_state_dict(torch.load("fruit_classifier.pth", map_location=device))
    model = model.to(device)
    model.eval()
    return model

def main():
    st.set_page_config(
        page_title="Fruit recognition", 
        page_icon="🍓", 
        layout="centered", 
        initial_sidebar_state="auto", 
        menu_items= {
            'About': "This application recognizes fruits from images, the model used is resnet18 and was trained with the fruits-360 dataset."
        }
    )
    class_names = open("classes.txt").read().splitlines()
    model = load_model()

    st.title("Fruit recognition")
    st.write("Please upload one or more images, and we will try to recognize the fruits in them!")

    st.sidebar.title("Instructions")
    st.sidebar.write("Upload one or more images, and we will try to recognize the fruits in them using a convolutional neural network.")
    
    st.sidebar.title("Authors")
    st.sidebar.write("This application was created by Nicolas Blanquez, Yasmine ... and Ayman ....")

    st.sidebar.title("Source code")
    st.sidebar.write("The source code for this application can be found on [GitHub](https://github.com/nicblqz/UPM_DL)")

    uploaded_file = st.file_uploader(
        "Choose one or more images...", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )
    
    if uploaded_file is not None:
        # Load the image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Classifying...")

        # Preprocess the image
        input_image = transform(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(input_image)
            _, predicted_class = torch.max(outputs, 1)
    
        # Display result
        predicted_label = class_names[predicted_class.item()]
        st.write(f"### 🎯 Prediction: **{predicted_label}**")



if __name__ == "__main__":
    main()