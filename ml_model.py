import cv2

video = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier("/home/siddhesh/programming/smile_detection/smile-selfie-capture-project/dataset/haarcascade_frontalface_default.xml")
smileCascade = cv2.CascadeClassifier("/home/siddhesh/programming/smile_detection/smile-selfie-capture-project/dataset/haarcascade_smile.xml")

cnt = 1  # Initialize counter for image filenames

while True:
    success, img = video.read()
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(grayImg, 1.1, 4)
    
    for x, y, w, h in faces:
        # Draw rectangle around detected faceUpdates  from the Last Evaluation : 
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
            print("/home/siddhesh/programming/smile_detection/selfie" + str(cnt) + " Saved")
            path = "" + str(cnt) + ".jpg"
            cv2.imwrite(path, img_with_smile)
            cnt += 1
            
            # Break out of the loop after detecting the first smile
            break
    
    cv2.imshow('live video', img)
    keyPressed = cv2.waitKey(1)
    if keyPressed & 0xFF == ord('q'):
        break

video.release()                                  
cv2.destroyAllWindows()
