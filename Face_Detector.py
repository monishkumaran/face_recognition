import cv2
from random import randrange

#load some pre-trained data on frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to detect faces in
#img = cv2.imread('cp.jpg')
#to capture video from webcam
webcam = cv2.VideoCapture(1)

# Iterate forever over frames
while True:
    # Read the current frame
    successful_frame_read, frame = webcam.read()

    # Must convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    #Draw rectangles around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 10) 

    cv2.imshow('Paul Diwakar Face Detector', frame )
    key = cv2.waitKey(1)

    # Stop if q key is pressed
    if key==81 or key==113:
        break
# Release the VideoCapture object
webcam.release()

print("code completed")
