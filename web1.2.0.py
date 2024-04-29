import cv2
import streamlit as st
import base64
import uuid
import os

def main():
    st.set_page_config(
        page_title="Smile Detection App",
        page_icon="😊",
        layout="centered",
        initial_sidebar_state="auto",
    )

    st.title("Smile Detection App")
    st.markdown(
        """
        <style>
        body {
            background-image: url('https://images.unsplash.com/photo-1531251445707-1f000e1e87d0');
            background-size: cover;
            background-repeat: no-repeat;
            font-family: Arial, sans-serif;
            color: #FFFFFF;
            margin: 0;
            padding: 0;
        }
        .title {
            color: #FF5733;
            text-align: center;
            font-size: 36px;
            margin-bottom: 30px;
        }
        .button-container {
            display: flex;
            justify-content: center;
            margin-bottom: 30px;
        }
        .button {
            background-color: #FF5733;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            font-size: 18px;
            cursor: pointer;
            margin: 0 10px;
            transition: background-color 0.3s ease;
        }
        .button:hover {
            background-color: #FF8C66;
        }
        .video-container {
            display: flex;
            justify-content: center;
            margin-bottom: 30px;
        }
        .video {
            width: 640px;
            height: auto;
            border: 3px solid #FF5733;
            border-radius: 10px;
        }
        .caption {
            text-align: center;
            font-size: 18px;
            margin-top: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Load Haar cascades
    faceCascade = cv2.CascadeClassifier("smile-selfie-capture-project/dataset/haarcascade_frontalface_default.xml")
    smileCascade = cv2.CascadeClassifier("smile-selfie-capture-project/dataset/haarcascade_smile.xml")

    # Check if cascades are loaded successfully
    if faceCascade.empty() or smileCascade.empty():
        st.error("Error loading cascade classifiers. Please check file paths.")
        return

    start_button_clicked = st.button("Start")

    if start_button_clicked:
        cnt = 1  # Initialize counter for image filenames

        # Capture video from webcam
        video = cv2.VideoCapture(0)

        while True:
            success, img = video.read()
            grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(grayImg, 1.1, 4)

            for x, y, w, h in faces:
                # Draw rectangle around detected face
                img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 3)

                roi_gray = grayImg[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]

                # Adjusted parameters for smile detection
                smiles = smileCascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=30, minSize=(50,50))

                if len(smiles) > 0:
                    # Smile detected, stop the video
                    filename = f"smile_{cnt}.jpg"
                    cv2.imwrite(filename, img)
                    download_link = create_download_link(filename, "Download Smile Image")
                    st.markdown(download_link, unsafe_allow_html=True)
                    cnt += 1
                    video.release()
                    return

            # Display video frame
            st.image(img, channels="BGR", caption="Live Video", use_column_width=True)

def create_download_link(filename, link_text):
    with open(filename, "rb") as file:
        encoded_file = base64.b64encode(file.read()).decode()
    href = f'<a href="data:file/jpg;base64,{encoded_file}" download="{filename}">{link_text}</a>'
    return href

if __name__ == "__main__":
    main()
