import cv2
import os
import numpy as np
from keras.models import load_model
from pygame import mixer
import streamlit as st  # Import Streamlit here

# Initialize the mixer for sound alerts
mixer.init()
sound = mixer.Sound('alarm.wav')

# Load the pre-trained Haar Cascade classifiers
face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

# Load the pre-trained model for eye detection
model = load_model('models/cnncat2.h5')
lbl = ['Close', 'Open']

# Function to detect drowsiness continuously
def detect_drowsiness(alert_message):
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    score = 0
    thicc = 2
    rpred = [99]
    lpred = [99]
    count = 0
    path = os.getcwd()

    while True:
        ret, frame = cap.read()
        height, width = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
        left_eye = leye.detectMultiScale(gray)
        right_eye = reye.detectMultiScale(gray)

        # Draw rectangle at the bottom of the frame (UI element)
        cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

        for (x, y, w, h) in right_eye:
            r_eye = frame[y:y + h, x:x + w]
            count += 1
            r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye, (24, 24))
            r_eye = r_eye / 255
            r_eye = r_eye.reshape(24, 24, -1)
            r_eye = np.expand_dims(r_eye, axis=0)
            rpred = model.predict(r_eye)
            rpred = np.argmax(rpred, axis=1)

            if rpred[0] == 1:
                lbl = 'Open'
            if rpred[0] == 0:
                lbl = 'Closed'
            break

        for (x, y, w, h) in left_eye:
            l_eye = frame[y:y + h, x:x + w]
            count += 1
            l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
            l_eye = cv2.resize(l_eye, (24, 24))
            l_eye = l_eye / 255
            l_eye = l_eye.reshape(24, 24, -1)
            l_eye = np.expand_dims(l_eye, axis=0)
            lpred = model.predict(l_eye)
            lpred = np.argmax(lpred, axis=1)

            if lpred[0] == 1:
                lbl = 'Open'
            if lpred[0] == 0:
                lbl = 'Closed'
            break

        # If both eyes are closed, increase the score
        if rpred[0] == 0 and lpred[0] == 0:
            score += 1
            cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            score -= 1
            cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        if score < 0:
            score = 0
        cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        # Take snapshot and show the frame only if drowsy
        if score > 15:
            # Person is feeling sleepy, beep the alarm and take a snapshot
            cv2.imwrite(os.path.join(path, 'image.jpg'), frame)  # Save the snapshot
            try:
                sound.play()  # Play the alarm sound
            except:
                pass
            if thicc < 16:
                thicc += 2
            else:
                thicc -= 2
                if thicc < 2:
                    thicc = 2
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
            
            # Display the frame in Streamlit and print alert message
            st.image(frame, channels="BGR", use_container_width=True)  # Show the frame when drowsy
            alert_message.warning("Drowsiness Detected! Please take a break.")  # Display alert when drowsy
        else:
            # Continue the loop without saving or displaying anything when not drowsy
            pass

        # Exit the loop when pressing the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
