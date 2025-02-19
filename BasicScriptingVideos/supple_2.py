import cv2

img = cv2.imread('DaBloat.png')
cv2.imshow("Face", img[150:750, 350:750])
cv2.waitKey()
cv2.destroyAllWindows()