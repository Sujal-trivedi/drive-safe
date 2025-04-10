# drive-safe
 An drowsiness detection and alert System
# Drowsiness Detection System
# LInk to dataset:- https://www.kaggle.com/datasets/serenaraju/yawn-eye-dataset-new
## Overview
This project implements a **Drowsiness Detection System** using a **Convolutional Neural Network (CNN)** for real-time eye status prediction. It detects if a person is drowsy based on their eye movements using the webcam feed. If drowsiness is detected, the system raises an alert through an **alarm sound** and **frame capture**.

This project is built using **OpenCV**, **TensorFlow** for deep learning model prediction, **pygame** for playing the alarm, and **Streamlit** for a user-friendly web interface.

## Features
- Real-time drowsiness detection based on eye status (open or closed).
- Continuous webcam feed with real-time processing of eye status.
- **Alarm sound** triggered when drowsiness is detected.
- **Frame snapshot** taken when drowsiness is detected.
- Web interface using **Streamlit** to interact with the system.
- Option to start and stop the detection process through the interface.
