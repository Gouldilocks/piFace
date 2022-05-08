# piFace, Christian Gould
A repository exploring facial expression recognition using openCV
- This is the final project for Applied Machine Learning with Dr. Fontenot, CS-5394
- This project was approved by Dr. Fontenot, as an alternative to the original represented in the handout. The task was this:
  - Explore using OpenCV with human features
  - Create some type of model to predict the emotion of a person with an image
  - use these two above points to create live- image prediction of a person's emotions

Dataset used to train the model: https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer

# Running the Project
* To run the project, simply type into your terminal command prompt:
```
python3 piface.py
```
* There may be some dependencies that you have to download. The libraries used to run the final product are:
  * tensorflow
  * matplotlib
  * numpy
  * opencv

* There are also subsidiary "projects" that were used for exploration, namely the files: "create_model.py", and "person_detector.py". Each of the three files will be talked about below.

### piface.py
* piface is the final product of for this project, which was the area I left off at in this journey. Originally, my goal was to put emotion detection using machine learning modelling into a raspberry pi, and make a personalized mirror like [this one](https://all3dp.com/2/raspberry-pi-magic-mirror-smart-best-project/) which would tell you a quote based on your mood. I did not quite get that far, but I made the proof of concept. 

* The program will load a previously trained model "model.h5", which was acquired from the "create_model.py" file. 

* After loading the model, it will cycle through a list of paths to images in the "images_to_predict" variable. Feel free to add more and see what it predicts. 

* Once it makes predictions on those images, it will show its confidence with a percentage, and then move on to live video

* Once the live video pops up, it will show your webcam, with a little red bar below your face, indicating your current emotion that it thinks you are exemplifying. Press "q" to quit the window. 

### create_model.py
* In this modelling script, I took portions from these previously-made models, and adjusted it to what I wanted:
  * https://www.kaggle.com/code/raksharamkumar/expdet
  * https://www.kaggle.com/code/ebrarteke/emotion-detection/notebook
  * https://www.kaggle.com/code/sairamankoraveni/final-face-emotion-recog

* This model took my computer about 12 hours to run, which may be the result of my computer being slow, or me doing something incorrectly, which made it very slow. In the end, the model was saved to "model.h5"

### person_detector.py
* In this script, I was exploring what openCV was capable of.
* I was able to make openCV take video frame by frame, and use a pre-trained model in order to identify between different people.
* In the future, I want to add the ability to have different emotion sets tailored to different people, using something like deepface.
  * [Link to Deepface example](https://medium.com/analytics-vidhya/human-face-emotion-and-race-detection-with-python-86ca573e0c45)

# Challenges / Lessons Learned
* OpenCV was a new thing to me, and I had never used it prior to the start of thish project. After using it and learning about it for this project, I think that I have a pretty decent grasp as to how to use it effectively for future projects I may use

* using Keras was new to me. When I was researching about making a model for detecting human emotion, I continued to come across examples using Tensorflow Keras. Because we had not used it in class, and the book did not give us examples of using Tensorflow, I was somewhat intimidated, but after enough trial and error, I was able to get it to model the way that I desired.

* Hardware problems. It turns out that there are some hardware issues when it comes to using M1 macs and some of these libraries. For instance, on M1, I am unable to use the same conda environment for some of the files. I have to use python 3.9 for creating the model, as well as for the person_detector. But, when it comes to using the whole piface program, I need to use python 3.7 because tensorflow does not work on M1 with python 3.9.
