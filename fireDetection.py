import numpy as np
import cv2

fire_cascade = cv2.CascadeClassifier('fire_detection.xml')

cap = cv2.VideoCapture(0)

while(True):
	ret, frame = cap.read()
	if not ret:
		break

	blur = cv2.GaussianBlur(frame, (21, 21), 0)
	hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	lower = np.array((18, 50, 50), dtype="uint8")
	upper = np.array((35, 255, 255), dtype="uint8")
	mask = cv2.inRange(hsv, lower, upper)

	output = cv2.bitwise_and(frame, hsv, mask=mask)
	no_red = cv2.countNonZero(mask)
	fire = fire_cascade.detectMultiScale(frame, 1.2, 5)

	cv2.imshow("output", output)
	if int(no_red) > 5000:
		print("Primary detection")
		for (x,y,w,h) in fire:
			cv2.rectangle(frame,(x-20,y-20),(x+w+20,y+h+20),(255,0,0),2)
			cv2.rectangle(output,(x-20, y-20),(x+w+20,y+h+20),(255,0,0),2)
			roi_gray = gray[y:y+h, x:x+w]
			roi_color = frame[y:y+h, x:x+w]
			print("Finally Fire Detect!")
		print("====================")

	cv2.imshow("input", frame)
	if cv2.waitKey(1) & 0xFF == 27:
		break

cap.release()
cv2.destroyAllWindows()
