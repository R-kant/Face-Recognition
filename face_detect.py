import cv2
import numpy as np
from os import path


name = input("Enter your Name : ")

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

count = 50
face_list = list()
while(True):

    ret , frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #gray = cv2.resize(gray, (200, 200))
    #cv2.imshow("picture", gray)

    faces = classifier.detectMultiScale(gray)

    if(len(faces)>0):


        face = faces[get_biggest_face(faces)]

        x, y, w, h = tuple(face)

        face = gray[y:y+h, x:x+w]

        face = cv2.resize(face, (100, 100))
        face_list.append(face.flatten())
        font = cv2.FONT_HERSHEY_SIMPLEX
        count-=1
        cv2.rectangle(frame, (x - 10, y - 10), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(frame, name, (x - 15, y - 15), font, 2, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Picture', frame)

    if(cv2.waitKey(1)>30 or count<1):
        break

face_list = np.array(face_list)
name_list = np.full((len(face_list), 1), name)
face_data = np.hstack([name_list, face_list])

if(path.exists("faces.npy")):

    data = np.load("faces.npy")
    data = np.vstack([data, face_data])

else:

    data = face_data

np.save("faces.npy", data)


cap.release()
cv2.destroyAllWindows()