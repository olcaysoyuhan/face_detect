import cv2 as cv
import face_recognition
import dlib
from datetime import datetime
import sqlite3
import os

path = "img/"
con = sqlite3.connect('db/employee_detect.db')
cursor = con.cursor()
detector = dlib.get_frontal_face_detector()
known_encodings = {}
known_names = {}

with os.scandir(path) as file:
    for entry in file:
        if entry.is_file():
            employee_name, _ = os.path.splitext(entry.name)
            photos = os.path.join(path, entry.name)
            photo = face_recognition.load_image_file(photos)
            photo_enc = face_recognition.face_encodings(photo)[0]
            known_encodings[employee_name] = photo_enc
            known_names[employee_name] = employee_name


def detect_table():
    cursor.execute('CREATE TABLE IF NOT EXISTS Employee(name TEXT, time DATETIME)')
    con.commit()


detect_table()


def detect_add(name, time):
    cursor.execute('INSERT INTO Employee VALUES(?,?)', (name, time))
    con.commit()


cap = cv.VideoCapture(0)

while cap.isOpened():
    _, frame = cap.read()
    frame = cv.flip(frame, 1)
    faces = detector(frame)
    face_loc = [(face.top(), face.right(), face.bottom(), face.left()) for face in faces]
    face_enc = face_recognition.face_encodings(frame, face_loc)
    for i, face in enumerate(face_enc):
        y, w, h, x = face_loc[i]
        results = face_recognition.compare_faces(list(known_encodings.values()), face)
        if True in results:
            matched_index = results.index(True)
            matched_name = known_names[list(known_encodings.keys())[matched_index]]
            cv.rectangle(frame, (x, y), (w, h), (255, 0, 0), 2)
            cv.putText(frame, matched_name,
                       (x, h + 35), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            detect_add(matched_name, datetime.now())
    cv.imshow('face detect', frame)
    if cv.waitKey(1) == 27:
        break
cap.release()
cv.destroyAllWindows()
con.close()
