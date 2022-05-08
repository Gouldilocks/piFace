import numpy as np
import cv2
import os
#  import matplotlib.pyplot as plt
# import seaborn as sns

video_capture = cv2.VideoCapture(0)
# loop over all files in directory "people"
known_face_encodings = []
known_face_names = []

# for filename in os.listdir("people"):
#     # remove the file extension
#     name = filename.split(".")[0]
#     print("name: ", name)

#     # load the image
#     image = fr.load_image_file("people/" + filename)
#     try:
#         known_face_encodings.append(fr.face_encodings(image)[0])
#         known_face_names.append(name)
#     except Exception as e:
#         print("error: ", e)
#         print("no face found in ", filename)
#         continue
 
while True:
    ret, frame = video_capture.read()

    # faces = detector(gray) # Used for the dots

    rgb_frame = frame[:, :, ::-1]
    gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY) # used for the dots

    # face_locations = fr.face_locations(rgb_frame)
    # face_encodings = fr.face_encodings(rgb_frame, face_locations)

    # for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    #     print("top: ", top)
    #     print("right: ", right)
    #     print("bottom: ", bottom)
    #     print("left: ", left)

    #     matches = fr.compare_faces(known_face_encodings, face_encoding)

    #     name = "Unknown Face"

    #     face_distances = fr.face_distance(known_face_encodings, face_encoding)

    #     best_match_index = np.argmin(face_distances)
    #     if matches[best_match_index]:
    #         name = known_face_names[best_match_index]

    #     cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    #     cv2.rectangle(frame, (left, bottom - 35),
    #                   (right, bottom), (0, 0, 255), cv2.FILLED)
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     cv2.putText(frame, name, (left + 6, bottom - 6),
    #                 font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Webcam_facerecognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
