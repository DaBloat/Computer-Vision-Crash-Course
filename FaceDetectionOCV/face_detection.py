import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
camera = cv2.VideoCapture(2)
  
def detect():
  """
  This Function Detects the feed Face and the Eyes
  """
  while (True):
    ret, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
      img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
      roi_gray = gray[y:y+h, x:x+w]
      eyes = eye_cascade.detectMultiScale(roi_gray)
    
      for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(img[y:y + h, x:x + w],(ex,ey),(ex+ew,ey+eh),
        (0,255,0),2)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("camera", frame)
    if cv2.waitKey(1) & 0xff == ord("q"):
      break
    
  camera.release()
  cv2.destroyAllWindows()
  
  
def detect_with_smile():
  """
  This function Detects the Face, Eyes and smile
  """
  while (True):
    ret, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
      img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
      roi_gray = gray[y:y+h, x:x+w]
      
      eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 22)
      for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(img[y:y + h, x:x + w],(ex,ey),(ex+ew,ey+eh),
        (0,255,0),2)
      
      
      smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
      for (sx,sy,sw,sh) in smiles:
        cv2.rectangle(img[y:y + h, x:x + w],(sx,sy),(sx+sw,sy+sh),
        (0,0,255),2)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("camera", frame)
    if cv2.waitKey(1) & 0xff == ord("q"):
      break
    
  camera.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_with_smile()