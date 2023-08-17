# Import opencv
import cv2 
# Import uuid
import uuid
import time
import os
import mediapipe as mp
import tensorflow as tf
import numpy as np

def mp_detection(image,model):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable=False
    results=model.process(image)
    image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    image.flags.writeable=True
    return image, results

mp_holistic=mp.solutions.holistic #Holistic model
#mp_drawing=mp.solutions.drawing_utils #Drawing Cosmitics

#Camara
#Mediapipe model
Name="Hemant"

mp_holistic=mp.solutions.holistic #Holistic model
mp_drawing=mp.solutions.drawing_utils #Drawing Cosmitics

data_dir="./Data/Raw_Images/"
os.makedirs(data_dir+Name+"/", exist_ok=True)

#  Image capture
cap = cv2.VideoCapture(0)
count=1
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:
        c=len(os.listdir(data_dir+Name+"/"))
        ret, frame = cap.read()
        image, results = mp_detection(frame, holistic)     
            # Draw landmarks
        #mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,mp_drawing.DrawingSpec(color=(255,0,255),thickness=1,circle_radius=1))
        cv2.putText(image,str(c),(30,30) ,cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        k = cv2.waitKey(50)
        cv2.imshow('frame', image)
        
        if (count):
            if(k%256 == 32):
                count=0
        else:
            k = cv2.waitKey(100)
            if k%256 == 27 or c==100:
                break
    
            if (results.face_landmarks):
                imgname = data_dir+Name+"/{}.jpg".format(str(uuid.uuid1()))
                print(imgname)
                cv2.imwrite(imgname, frame)

cap.release()
cv2.destroyAllWindows()
