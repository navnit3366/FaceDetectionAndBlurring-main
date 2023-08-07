import numpy as np
import cv2

# Loading the classifier for frontal face:
FaceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Start video capturing using the embedded camera (built-in webcam):
capture = cv2.VideoCapture(0)

# Output video meta data and format:
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
video_output = cv2.VideoWriter('ResultVideo.avi', fourcc, 5.0, (640, 480))

# Main program: (press 'q' or 'Q' to quit)
while(True):
    # input video:
    _, video_input = capture.read()

    # Change the original video into a gray-scaled video:
    video_gray = cv2.cvtColor(video_input, cv2.COLOR_BGR2GRAY)

    # detect faces in input video (after gray scaling):
    faces = FaceDetector.detectMultiScale(video_gray, 1.2, 5)

    # Print out the number of found faces:
    print(f"Faces found: {len(faces)} \t You can quit the program by pressing 'q' or 'Q'")

    # now we have the (x, y) and (width, height) of all faces in input video.
    # loop to draw a Rectangle around all faces , and put the filter around it:
    for (x,y,w,h) in faces:
        # (blue, green, red) => (255, 0, 0)
        # Draw a rectangle with blue outlines:
        cv2.rectangle(video_input, (x,y), (x+w,y+h), (255, 0, 0), 2)

        # Copy a found face from the video:
        video_blur = video_input[y:y+h, x:x+w]

        # Apply Filter on the detected face to blur it:
        video_blur = cv2.GaussianBlur(video_blur, (31,31), 0)

        # Paste the blurred face back to its original place:
        video_input[y:y+h, x:x+w] = video_blur

    # Display the video after blurring faces:
    video_output.write(video_input)
    cv2.imshow('Live Video', video_input)

    # to QUIT the program: >>> press (q) OR (Q) button
    key = cv2.waitKey(1) & 0xFF
    if (key == ord('q')) | (key == ord('Q')):
        break

# Close any open windows:
video_output.release()
capture.release()
cv2.destroyAllWindows()
