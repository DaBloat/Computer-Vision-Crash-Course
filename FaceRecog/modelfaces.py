import cv2
import numpy as np
from load_data import read_images

eigen = cv2.face.EigenFaceRecognizer_create()
fisher = cv2.face.EigenFaceRecognizer_create()
lbph = cv2.face.EigenFaceRecognizer_create()

def face_rec(model):
  names = ['DaBloat', 'Keanna'] # Put your names here for faces to recognize

  [X, y] = read_images('Faces', sz=(500, 500))
  y = np.asarray(y, dtype=np.int32)

  model.train(X, y)

  camera = cv2.VideoCapture(2)
  face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

  while True:
    ret, img = camera.read()
    if not ret:
      break

    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    for (x, y, w, h) in faces:
      cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
      gray = cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
      roi = cv2.resize(gray, (200, 200), interpolation=cv2.INTER_LINEAR)

      try:
        params = model.predict(roi)
        label = names[params[0]]
        cv2.putText(img, label + ", " + str(params[1]), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
      except:
        continue

    cv2.imshow("camera", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
      break

  camera.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  face_rec(lbph)