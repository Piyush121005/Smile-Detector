for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (w+x,h+y), (0,0,240),3)

        # Get the sub frame (using Numpy N Dimensional Array)
        the_face = frame[y:y+h, x:x+w]

        face_grayscaled = cv2.cvtColor(the_face,cv2.COLOR_BGR2GRAY)

        smile_coordinates = smile_detector.detectMultiScale(face_grayscaled, scaleFactor=1.8, minNeighbors=20)

        # Draw Rectangle around the smile
        for (x_, y_, w_, h_) in smile_coordinates:
            cv2.rectangle(the_face, (x_, y_), (x_ + w_, y_ + h_), (120,220,0),1)