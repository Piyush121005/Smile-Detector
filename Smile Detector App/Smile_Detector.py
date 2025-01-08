import cv2

# Load some pre-trained data on face fronal and smile
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smile_detector = cv2.CascadeClassifier("haarcascade_smile.xml")

# To Capture video from webcam
webcam = cv2.VideoCapture(0)

# Iterate forever over frames
while True:

    # Read the current frame
    # succesful_frame_read is a boolean and frame is the actual image
    successful_frame_read, frame = webcam.read()

    if not successful_frame_read:
        break

    # Must convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect Face and smile
    face_coordinates = face_detector.detectMultiScale(grayscaled_img, scaleFactor=1.1 , minNeighbors=20)

    # Drawing Rectangle for face and smile
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (w+x,h+y), (40,0,177),3)

        # Get the sub frame (using Numpy N Dimensional Array)
        the_face = frame[y:y+h, x:x+w]

        face_grayscaled = cv2.cvtColor(the_face,cv2.COLOR_BGR2GRAY)

        smile_coordinates = smile_detector.detectMultiScale(face_grayscaled, scaleFactor=1.8, minNeighbors=20)

        # Label the rectangle as smiling when the smile is detected in the face
        if len(smile_coordinates)>0:
            cv2.putText(frame, 'Smiling', (x+5, y+h+40), fontScale=1.2, fontFace = cv2.FONT_HERSHEY_DUPLEX, color = (190,222,181))
            

        


    cv2.imshow("Bhavya Piyush Smile Detecor", frame)

        # It pauses the execution of the code
    # Automatically press a key after 1 millisecond
    key = cv2.waitKey(1)

    # Upper and lower case ASCII value of Q
    if key == 81 or key == 113:
        break

# Cleanup

webcam.release()
cv2.destroyAllWindows()

print("Code Completed!")