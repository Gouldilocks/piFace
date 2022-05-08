import numpy as np
import face_recognition as fr
import cv2
import os
import dlib
import matplotlib.pyplot as plt
import seaborn as sns

# # Importing Deep Learning Libraries
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
# from tensorflow.keras.models import Model, Sequential
# from tensorflow.keras.optimizers import Adam, SGD, RMSprop
print("hello")
picture_size = 48
folder_path = "./data/images"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")

video_capture = cv2.VideoCapture(0)
# loop over all files in directory "people"
known_face_encodings = []
known_face_names = []

for filename in os.listdir("people"):
    # remove the file extension
    name = filename.split(".")[0]
    print("name: ", name)

    # load the image
    image = fr.load_image_file("people/" + filename)
    try:
        known_face_encodings.append(fr.face_encodings(image)[0])
        known_face_names.append(name)
    except Exception as e:
        print("error: ", e)
        print("no face found in ", filename)
        continue
 
while True:
    ret, frame = video_capture.read()

    # gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY) # used for the dots
    # faces = detector(gray) # Used for the dots

    rgb_frame = frame[:, :, ::-1]

    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        print("top: ", top)
        print("right: ", right)
        print("bottom: ", bottom)
        print("left: ", left)

        # for face in faces:
        #     x1 = face.left()  # left point
        #     y1 = face.top()  # top point
        #     x2 = face.right()  # right point
        #     y2 = face.bottom()  # bottom point

        #     # Look for the landmarks
        #     landmarks = predictor(image=gray, box=face)

        #     for n in range(0, 68):
        #         x = landmarks.part(n).x
        #         y = landmarks.part(n).y

        #         # Draw a circle
        #         cv2.circle(img=frame, center=(x, y), radius=2,
        #                 color=(0, 255, 0), thickness=-1)
 
        matches = fr.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown Face"

        face_distances = fr.face_distance(known_face_encodings, face_encoding)

        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 35),
                      (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Webcam_facerecognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
