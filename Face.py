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

model_dir="./Model/"
model_file=os.listdir(model_dir)
model_path=model_dir+model_file[0]
model=tf.keras.models.load_model(model_path)
labels=os.listdir("./Data/Raw_Images/")

#Mediapipe model
mp_holistic=mp.solutions.holistic #Holistic model
#mp_drawing=mp.solutions.drawing_utils #Drawing Cosmitics
flag=0
count=0
#  Image capture
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:
        if(flag):
            count=count + 1
            count=count % len(model_file)
            model_path=model_dir+model_file[count]
            model=tf.keras.models.load_model(model_path)
            flag=0
            print("Model changed")

        ret, frame = cap.read()
        image, results = mp_detection(frame, holistic)  
        #mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,mp_drawing.DrawingSpec(color=(255,0,255),thickness=1,circle_radius=1))   

            # Draw landmarks
        if(not results.face_landmarks):
            cv2.putText(image, "No Face Detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)    
        
        else:
            img=cv2.resize(frame,(640, 480))
            #img=load_and_prep_image(frame,img_shape=224)
            pred = model.predict(tf.expand_dims(img, axis=0),verbose=0)
            pred_class = labels[pred.argmax()]
            # print(labels)
            # print(pred_class)
            # print(pred)
            #cv2.putText(image, "Unreconized Face", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image,pred_class+" "+str(pred.max()),(30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image,"Model: "+str(model_file[count]), (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("Frame",image)
        k = cv2.waitKey(1)

        if(k%256 == 32):
            flag=1
        if k%256 == 27:
            break

cap.release()
cv2.destroyAllWindows()