import cv2
import streamlit as st
import base64
import uuid

def main():
    st.title("Smile Detection App")
    st.markdown(
        """
        <style>
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

                if len(smiles) == 0:
                    # No smile detected, prompt the user to come closer
                    cv2.putText(img, "Please come closer", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    # Smile detected, proceed with smile detection
                    x1, y1, w1, h1 = smiles[0]  # Take the first detected smile
                    img_with_smile = cv2.rectangle(roi_color, (x1, y1), (x1+w1, y1+h1), (100, 100, 100), 5)
                    st.image(img_with_smile, channels="BGR", caption="Smile Detected")
                    cnt += 1

                    # Break out of the loop after detecting the first smile
                    break

            st.markdown('<div class="video-container"><img src="data:image/png;base64,{}" class="video"></div>'.format(img_to_base64(img)), unsafe_allow_html=True)
            stop_button_clicked = st.button("Stop", key=str(uuid.uuid4()))  # Generate a unique key
            if stop_button_clicked:
                break

        video.release()

def img_to_base64(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode()

if __name__ == "__main__":
    main()
