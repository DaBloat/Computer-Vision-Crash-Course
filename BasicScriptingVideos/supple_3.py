import cv2

record = False

cap = cv2.VideoCapture(0)

if (cap.isOpened() == False): 
  print("Unable to read camera feed")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

while(True):
  ret, frame = cap.read()
  k = cv2.waitKey(1)

  if ret == True: 
    cv2.imshow('frame',frame)

    # press space key to start recording
    if k%256 == 32:
        record = True

    if record:
        out.write(frame) 

    # press q key to close the program
    if k & 0xFF == ord('q'):
        break

  else:
     break  

cap.release()
out.release()

cv2.destroyAllWindows()