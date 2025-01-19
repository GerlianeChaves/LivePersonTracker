#imports
import cv2

#webcam
cam = cv2.VideoCapture(0) #0 to webcam

while True:
    ret, frame = cam.read()

    if not ret:
        break
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()