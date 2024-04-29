from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Load pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained deep learning model for smile detection
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Function to detect smiles
def detect_smile(image_data):
    # Convert base64 string to numpy array, etc.
    nparr = np.frombuffer(base64.b64decode(image_data.split(',')[1]), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Copy the image before drawing rectangles
    img_for_download = img.copy()

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    smile_detected = False

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=30, minSize=(50,50))

        if len(smiles) > 0:
            smile_detected = True
            # Only draw rectangles on the display image, not on the download image
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(img, (x+sx, y+sy), (x+sx+sw, y+sy+sh), (0, 255, 0), 2)

    # Encode the download image without rectangles
    _, img_encoded = cv2.imencode('.jpg', img_for_download)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    return img_base64, smile_detected

# Route to render HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image processing
@app.route('/detect_smile', methods=['POST'])
def process_image():
    image_data = request.form['image']
    img_base64, smile_detected = detect_smile(image_data)
    if smile_detected:
        return jsonify({'result': 'smile_detected', 'image': img_base64})
    else:
        return jsonify({'result': 'no_smile'})

if __name__ == "__main__":
    app.run(debug=True)
