import cv2

cap = cv2.VideoCapture('test1.mp4')

if (cap.isOpened()== False):
    print("Error opening video file")

while(cap.isOpened()):
    
    ret, frame = cap.read()
    if ret == True:
    # Display the resulting frame
        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()