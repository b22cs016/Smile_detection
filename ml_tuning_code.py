import cv2
import time

video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error: Failed to open video capture.")
    exit()

faceCascade = cv2.CascadeClassifier("/home/siddhesh/programming/smile_detection_version1.0/smile-selfie-capture-project/dataset/haarcascade_frontalface_default.xml")
smileCascade = cv2.CascadeClassifier("/home/siddhesh/programming/smile_detection_version1.0/smile-selfie-capture-project/dataset/haarcascade_smile.xml")

# Define ranges for parameter tuning
scale_factors = [1.1, 1.2, 1.3, 1.4, 1.5]
min_neighbors_values = [3, 5, 10, 20, 30]
min_sizes = [(30, 30), (40, 40), (50, 50), (60, 60)]

# Initialize variables to store parameters for which smile is detected
detected_parameters = []

for scale_factor in scale_factors:
    for min_neighbors in min_neighbors_values:
        for min_size in min_sizes:
            print(f"Testing parameters: scaleFactor={scale_factor}, minNeighbors={min_neighbors}, minSize={min_size}")
            
            cnt = 1  # Initialize counter for image filenames
            smile_detected = False  # Flag to track if smile has been detected
            start_time = time.time()  # Start time for smile detection

            while time.time() - start_time < 2:  # Check for smile detection within 2 seconds
                success, img = video.read()
                if not success:
                    print("Error: Failed to read frame from the video capture.")
                    break

                grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale(grayImg, 1.1, 4)

                for x, y, w, h in faces:
                    img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 3)

                    roi_gray = grayImg[y:y+h, x:x+w]
                    roi_color = img[y:y+h, x:x+w]

                    smiles = smileCascade.detectMultiScale(roi_gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size)

                    if len(smiles) > 0:
                        smile_detected = True
                        break  # Exit the loop if smile is detected

                cv2.imshow('live video', img)
                keyPressed = cv2.waitKey(1)
                if keyPressed & 0xFF == ord('q') or smile_detected:
                    break
            
            if smile_detected:
                detected_parameters.append((scale_factor, min_neighbors, min_size))
                break  # Move on to the next set of parameters

# Calculate the average of parameters for which smile is detected
if detected_parameters:
    avg_scale_factor = sum(p[0] for p in detected_parameters) / len(detected_parameters)
    avg_min_neighbors = sum(p[1] for p in detected_parameters) / len(detected_parameters)
    avg_min_size = tuple(sum(x) / len(detected_parameters) for x in zip(*[p[2] for p in detected_parameters]))

    print("Optimal Parameters:")
    print(f"Average scaleFactor: {avg_scale_factor}")
    print(f"Average minNeighbors: {avg_min_neighbors}")
    print(f"Average minSize: {avg_min_size}")
else:
    print("No smile detected for any parameter combination.")

# Release the video capture before exiting
video.release()                                  
cv2.destroyAllWindows()
