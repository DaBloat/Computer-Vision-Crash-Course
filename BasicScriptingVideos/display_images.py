import cv2
import numpy as np

img = cv2.imread('DaBloat.png')
cv2.imshow('DaBloat', img)
cv2.waitKey()
cv2.destroyAllWindows()