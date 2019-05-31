import numpy as np
import cv2

fire_cascade = cv2.CascadeClassifier('firecomp.xml')

cap = cv2.VideoCapture(0)

while True:
	(grabbed, img) = cap.read()
	if not grabbed:
		break

	blur = cv2.GaussianBlur(img, (21, 21), 0)
	hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

	lower = [18, 50, 50]
	upper = [35, 255, 255]
	lower = np.array(lower, dtype="uint8")
	upper = np.array(upper, dtype="uint8")
	mask = cv2.inRange(hsv, lower, upper)

	output = cv2. bitwise_and(img, hsv, mask=mask)
	no_red = cv2.countNonZero(mask)

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	fire = fire_cascade.detectMultiScale(gray, 50, 50)

	for (x,y,w,h) in fire:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)

		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]

		if int(no_red) > 20000:
			print('Fire detected')

	cv2.imshow("output", output)
	cv2.imshow('input',img)

	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()
