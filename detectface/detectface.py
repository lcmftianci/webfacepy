import numpy as np
import cv2
import matplotlib.pyplot as plot

face_cascade = cv2.CascadeClassifier("C:\\Opencv3.2.0\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt2.xml")
eye_cascade = cv2.CascadeClassifier("C:\\Opencv3.2.0\\opencv\\sources\\data\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml")

# img = cv2.imread("D:\\Databases\\picdata\\girls.jpeg")

def cvdetect(myframe):
	gray = cv2.cvtColor(myframe, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray,1.1,5,cv2.CASCADE_SCALE_IMAGE)
	if len(faces) > 0:
		for faceRect in faces:
			x,y,w,h = faceRect
			cv2.rectangle(myframe, (x, y), (x + w, y + h), (255, 0, 0), 2, 8, 0)
			roi_gray = gray[y:y+h, x:x+w]
			roi_color = myframe[y:y+h, x:x+h]
			eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 1, cv2.CASCADE_SCALE_IMAGE, (2,2))
			for(ex,ey,ew,eh) in eyes:
				cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0),2)
	cv2.imshow("img",myframe)


cap = cv2.VideoCapture(0)
while(1):
	ret,frame = cap.read()
# 	cv2.imshow("frame", frame)
	if cv2.waitKey(1)&0xFF==ord('q'):
		break
	cvdetect(frame)

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(0)

