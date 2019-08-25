import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier

data = np.load('faces.npy')

X = data[:, 1:].astype(int)
y = data[:, 0]


model = KNeighborsClassifier(4)

model.fit(X,y)


def get_biggest_face(data):
    area = list()
    i = 0
    for (x, y, w, h) in data:
        area.append(((w * h), i))
        i += 1

    area = sorted(area, reverse=True)

    return area[0][1]


cap = cv2.VideoCapture(0)

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while(True):

    ret , frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


    faces = classifier.detectMultiScale(gray)

    if(len(faces)>0):


        face = faces[get_biggest_face(faces)]

        x, y, w, h = tuple(face)

        face = gray[y:y+h, x:x+w]

        face = cv2.resize(face, (100, 100))

        font = cv2.FONT_HERSHEY_SIMPLEX

        face = face.flatten()

        prediction = model.predict([face])
        print(prediction)
        cv2.rectangle(frame, (x - 10, y - 10), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(frame, prediction[0], (x - 15, y - 15), font, 2, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Picture', frame)

    if(cv2.waitKey(1)>30):
        break
cap.release()
cv2.destroyAllWindows()