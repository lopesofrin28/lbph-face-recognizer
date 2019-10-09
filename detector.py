import cv2
import os
import numpy as np


faceDetect=cv2.CascadeClassifier('C:\\Users\\kevin\\AppData\\Local\\Programs\\Python\\Python37\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml');
cam=cv2.VideoCapture(0);
#rec=cv2.createLBPHFaceRecognizer();
#rec=cv2.face.createLBPHFaceRecognizer();
rec=cv2.face.LBPHFaceRecognizer_create();
rec.read("recognizer\\trainningData.yml")
id=0

#font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4)
font = cv2.FONT_HERSHEY_SIMPLEX
#cv.putText(img,'OpenCV',(10,500), font, 4,(0,0,255),2,cv.LINE_AA)


while(True):
     ret,img=cam.read();
     gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
     faces=faceDetect.detectMultiScale(gray,1.3,5);
     for(x,y,w,h) in faces:
         cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
         id,conf=rec.predict(gray[y:y+h,x:x+w])
         if(id==1):
             id="Ofrin"
         elif(id==2):
             id="Amber"
         elif(id==3):
             id="monali"
         elif(id==4):
             id="chris"
         elif(id==5):
             id="linnet"
         else:
             id="Unknown"
         cv2.putText(img,str(id),(x,y+h),font,1,(0,0,255),2,cv2.LINE_AA);
       #  cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)

     cv2.imshow("Face",img);
     if(cv2.waitKey(1)==ord('q')):
         break;
cam.release()
cv2.destroyAllWindows()
