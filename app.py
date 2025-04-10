import streamlit as st
from drowsiness_detection import detect_drowsiness

# Streamlit UI
st.title('Drowsiness Detection System')

# Add a placeholder for the alert message
alert_message = st.empty()

# Start/stop detection
start_button = st.button("Start Detection")
stop_button = st.button("Stop Detection")

if start_button:
    st.write("Detection started!")

    # Call the imported function to start the detection process
    frame, is_drowsy = detect_drowsiness(alert_message)  # Pass the placeholder to display messages

    # Show the frame
    st.image(frame, channels="BGR", use_container_width=True)

    if is_drowsy:
        alert_message.warning("Drowsiness Detected! Please take a break.")  # Display alert when drowsy
    else:
        alert_message.success("No Drowsiness Detected! Keep going.")  # Display message when not drowsy

elif stop_button:
    st.write("Detection stopped!")
    alert_message.empty()  # Clear the alert message when detection is stopped
