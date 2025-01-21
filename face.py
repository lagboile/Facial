import cv2
import streamlit as st
import numpy as np
from PIL import Image

cascade_path = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    b = int(hex_color[4:6], 16)
    g = int(hex_color[2:4], 16)
    r = int(hex_color[0:2], 16)
    return (b,g,r)

def detect_faces(scale_factor, min_neighbors, color1):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Erreur lors de l'ouverture de la webcam.")
        return None

    ret, frame = cap.read()
    if not ret:
        st.error("Erreur lors de la capture de l'image.")
        cap.release()
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

    bgr_color = hex_to_bgr(color1)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), bgr_color, 2)

    cap.release()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame_rgb

def app():
    st.title("Détection de Visages avec l'Algorithme Viola-Jones")

    st.write("### Instructions :")
    st.write("1. Appuyez sur le bouton ci-dessous pour commencer à détecter des visages à partir de votre webcam.")
    st.write("2. Vous pouvez choisir la couleur des rectangles autour des visages détectés.")
    st.write("3. Ajustez les paramètres `minNeighbors` et `scaleFactor` pour personnaliser la détection.")
    st.write("4. Cliquez sur 'Enregistrer l'image' pour sauvegarder l'image avec les visages détectés.")

    color = st.color_picker("Choisissez la couleur des rectangles", "#00FF00")

    min_neighbors = st.slider("Ajustez minNeighbors", 1, 10, 5)
    scale_factor = st.slider("Ajustez scaleFactor", 1.1, 2.0, 1.3)

    if st.button("Détecter les Visages"):
        result_image = detect_faces(scale_factor, min_neighbors, color)

        if result_image is not None:

            st.image(result_image, caption="Image avec Visages Détectés", use_column_width=True)

            if st.button("Enregistrer l'image"):
                cv2.imwrite('detected_faces.png', cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
                st.success("Image enregistrée sous 'detected_faces.png'.")

if __name__ == "__main__":
    app()
