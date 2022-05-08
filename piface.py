from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
import numpy as np
import cv2


batch_size = 128
emotions = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# load the model for predicting emotion of a person
model = load_model('model.h5')

print()
print("loading model...")
print()

images_to_predict = ['./data/images/images/train/fear/2.jpg', './data/images/images/train/angry/22.jpg']

for img in images_to_predict:
  #load the image
  my_image = load_img(img, target_size=(48, 48), color_mode="grayscale")

  #preprocess the image
  my_image = img_to_array(my_image)
  my_image = my_image.reshape(
      (1, my_image.shape[0], my_image.shape[1], my_image.shape[2]))
  input_arr = np.array([my_image])
  #print the prediction
  prediction = model.predict(my_image)

  highest_prob = 0
  highest = 0
  index = 0
  for pred in prediction[0]:
      if pred > highest_prob:
          highest = index
          highest_prob = pred
      index+=1

  print("The predicted emotion for ", img, " is: ", emotions[highest], " with a probability of ", (highest_prob.round(3) * 100).round(2), "%")


# #####################################################################################################################
# Now to put the model in action for a live webcam feed
webCam = 0
for i in range(100):
  vid = cv2.VideoCapture(webCam)
  ret, frame = vid.read()
  if ret:
    webCam = i
    break

video_capture = cv2.VideoCapture(webCam)
# Load the cascade
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')

while True:
  try:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
      # take a 48x48 face image from the center of the face
      gray = gray[y:y+h, x:x+w]
      gray = cv2.resize(gray, (48, 48))
      gray = img_to_array(gray)
      img = gray.reshape((1, gray.shape[0], gray.shape[1], 1))
      input_arr = np.array([img])
      pred = model.predict(img)

      highest_prob = 0
      highest = 0
      index = 0
      for predi in pred[0]:
          if predi > highest_prob:
              highest = index
              highest_prob = predi
          index += 1

      print("The predicted emotion is: ", emotions[highest], " with a probability of ", (highest_prob.round(3) * 100).round(2), "%")
      cv2.rectangle(frame, (x, y+h - 35),
                    (x+w, y+h), (0, 0, 255), cv2.FILLED)
      cv2.putText(frame, "Emotion: " + emotions[highest], (x + 6, y+h - 6),
                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

      font = cv2.FONT_HERSHEY_SIMPLEX

        # Display the resulting frame
    cv2.imshow('Press q to quit', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  except:
    pass



