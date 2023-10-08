# Import necessary libraries
import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.utils import img_to_array
from playsound import playsound
from threading import Thread

# Function to play the alarm sound
def activate_alarm(sound_file):
    playsound(sound_file)

# Function to detect the status of eyes using a specific eye cascade and a trained model
def get_eye_status(eye_cascade, roi_color, model):
    # Resize the eye region to match the model's input size
    eye = cv2.resize(roi_color, (145, 145))
    # Normalize pixel values to the range [0, 1]
    eye = eye.astype('float') / 255.0
    # Convert to array and add an extra dimension
    eye = img_to_array(eye)
    eye = np.expand_dims(eye, axis=0)
    # Make a prediction using the model
    pred = model.predict(eye)
    # Return the index of the maximum value in the prediction array
    return np.argmax(pred)

# Main function
def main():
    # Define classes for eye status
    eye_status_classes = ['Closed', 'Open']
    
    # Load Haar cascades for face and eyes
    face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
    left_eye_cascade = cv2.CascadeClassifier("data/haarcascade_lefteye_2splits.xml")
    right_eye_cascade = cv2.CascadeClassifier("data/haarcascade_righteye_2splits.xml")
    
    # Initialize video capture
    video_capture = cv2.VideoCapture(0)
    
    # Load the pre-trained model for detecting sleepiness
    sleepiness_model = load_model("Sleep_Detection_Model.h5")
    
    # Initialize variables
    blinks_count = 0
    alarm_active = False
    alarm_sound_file = "data/alarm.mp3"
    left_eye_status = ''
    right_eye_status = ''

    # Main loop
    while True:
        # Read a frame from the video capture
        _, frame = video_capture.read()
        height, width, _ = frame.shape
        center_coordinates = (width // 2, height // 2)
        
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale frame
        detected_faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

        # Loop over each detected face
        for (x, y, w, h) in detected_faces:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
            
            # Extract the region of interest (ROI) for eyes
            roi_gray = gray_frame[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # Detect the status of the left eye
            left_eye_status = get_eye_status(left_eye_cascade, roi_color, sleepiness_model)
            
            # Detect the status of the right eye
            right_eye_status = get_eye_status(right_eye_cascade, roi_color, sleepiness_model)

            # Calculate the center of the rectangle
            center_x = x + w // 2
            center_y = y + h // 2

            # Check if both eyes are closed
            if left_eye_status == 2 and right_eye_status == 2:
                blinks_count += 1

                # Calculate the position for the "Eyes Closed" label
                text_closed = "Eyes Closed, Blink Count: " + str(blinks_count)
                text_size_closed = cv2.getTextSize(text_closed, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x_closed = center_x - text_size_closed[0] // 2
                text_y_closed = center_y - h // 2 - text_size_closed[1]  # Place the text above the rectangle

                cv2.putText(frame, text_closed, (text_x_closed, text_y_closed), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                # If eyes are closed for 1 consecutive frames, trigger an alarm
                if blinks_count >= 1:
                    # Display Text Label for Sleepiness Alert above the rectangle
                    text_alert = "Sleepiness Alert!!!"
                    text_size_alert = cv2.getTextSize(text_alert, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    text_x_alert = center_x - text_size_alert[0] // 2
                    text_y_alert = center_y + h // 2 + text_size_alert[1] // 2
                    cv2.putText(frame, text_alert, (text_x_alert, text_y_alert), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                    if not alarm_active:
                        alarm_active = True
                        # Play the alarm sound in a new thread
                        t = Thread(target=activate_alarm, args=(alarm_sound_file,))
                        t.daemon = True
                        t.start()
            else:
                # Display Text Label for Open Eyes above the rectangle
                text_open = "Eyes Open"
                text_size_open = cv2.getTextSize(text_open, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x_open = center_x - text_size_open[0] // 2
                text_y_open = center_y - h // 2 - text_size_open[1]  # Place the text above the rectangle
                cv2.putText(frame, text_open, (text_x_open, text_y_open), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                blinks_count = 0
                alarm_active = False

        # Display the frame
        cv2.imshow("Sleep Detector", frame)

        # Check for the 'q' key to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    video_capture.release()
    cv2.destroyAllWindows()

# Run the main function if this script is executed
if __name__ == "__main__":
    main()
