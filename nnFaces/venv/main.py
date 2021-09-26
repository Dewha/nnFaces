import tkinter as tk
from tkinter import ttk
from tkinter import *
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from PIL import ImageTk, Image, ImageFont, ImageDraw

import datetime
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import pickle
import csv
import pandas as pd
import pyodbc as db

# variables
isDetecting, confirm = False, False
makemodal = (len(sys.argv) > 1)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

id = 0
names = []
current_face = ()

win, cb_facility, current_date, \
current_time, txt_surname, txt_name, \
txt_secname, cb_post, cb_face, table, \
table_data, txt_date, txt_time, cb_facility_, label_visitor_info = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0

minW, minH = 0, 0

# DB
try:
    con_string = r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=.\FacesDB.mdb;'
    conn = db.connect(con_string)
    print("Connected To Database")
    cur = conn.cursor()
    cur.execute('SELECT * FROM Faces')
    for row in cur.fetchall():
        names.append((row[0], row[1], row[2], row[3], row[4]))
except db.Error as e:
    print("Error in Connection", e)

# метод определяющий маску на лице
def detect_and_predict_mask(frame, faceNet, maskNet, args):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces = []
    locs = []
    preds = []
    for i in range(0, detections.shape[1]):
        confidence = detections[0, 0, i, 2]
        if confidence > args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = frame[startY:endY, startX:endX]
            if len(face) > 0:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                faces.append(face)
                locs.append((startX, startY, endX, endY))
        break
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
    return (locs, preds, faces)

# метод включающий и выключающий определение личности
def detectPerson():
    global isDetecting, minH, minW, font, id, current_face, lbl_info, confirm
    if not isDetecting:
        isDetecting = True
        btn_start.configure(text='Завершиить')
        lbl_info.configure(text='Определение личности')
        root.update()
        ap = argparse.ArgumentParser()
        ap.add_argument("-f", "--face", type=str,
                        default="face_detector",
                        help="path to face detector model directory")
        ap.add_argument("-m", "--model", type=str,
                        default="mask_detector.model",
                        help="path to trained face mask detector model")
        ap.add_argument("-c", "--confidence", type=float, default=0.5,
                        help="minimum probability to filter weak detections")
        args = vars(ap.parse_args())
        print("[INFO] loading face detector model...")
        prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
        weightsPath = os.path.sep.join([args["face"],
                                        "res10_300x300_ssd_iter_140000.caffemodel"])
        faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
        print("[INFO] loading face mask detector model...")
        maskNet = load_model(args["model"])
        print("[INFO] starting video stream...")
        vs = cv2.VideoCapture(0)
        minW = 0.1 * vs.get(3)
        minH = 0.1 * vs.get(4)
        time.sleep(2.0)
        delay = 0

        # основной цикл работы
        while True:
            start = datetime.datetime.now()
            # считывание кадра с камары
            ret, frame = vs.read()
            frame = cv2.flip(frame, 1)
            # определение маски
            (locs, preds, faces_) = detect_and_predict_mask(frame, faceNet, maskNet, args)
            if len(locs) == 0: btn_toDB.configure(state='disabled')
            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred
                if mask > withoutMask:
                    # если обнаружена маска
                    label = "Пожалуйста снимите маску"
                    label_decision = "Не пропускать!"
                    label_visitor_info = ""
                    color_txt = '#FF0000'
                    color_txt_decision = '#FF0000'
                    color_face = (0, 0, 255, 0)
                    btn_toDB.configure(state='disabled')
                else:
                    # если маска не обнаружена то определение личности
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    face_name = 'Неизвестный'
                    label_decision = "Не пропускать!"
                    label_visitor_info = "Пожалуйста предъявите паспорт"
                    color_txt_decision = '#FF0000'
                    id, confidence = recognizer.predict(gray[startY:endY, startX:endX])
                    if (confidence < 100):
                        for row in names:
                            if row[0] == id:
                                face_name = row[1]+' '+row[2][0]+'.'+row[3][0]+'., '+ row[4]
                                current_face = row
                                label_decision = "Пропустить"
                                label_visitor_info = ""
                                color_txt_decision = '#00FF00'
                    else:
                        face_name = "Неизвестный"
                        label_decision = "Не пропускать!"
                        label_visitor_info = "Пожалуйста предьявите паспорт"
                        color_txt_decision = '#FF0000'
                    btn_toDB.configure(state='normal' if id != 'unknown' else 'disabled')
                    label = str(face_name)
                    color_txt = '#FFFFFF'
                    color_face = (0, 255, 0, 0)
                    break
            # надписи и рамки
            if len(locs) > 0:
                (startX, startY, endX, endY) = locs[0]
                canvas.delete('all')
                cv2.rectangle(frame, (startX, startY), (endX, endY), color_face, 3)
                frame = imutils.resize(frame, width=824)
                cv2.imwrite('img.jpg', frame)
                pilimage = Image.open('img.jpg')
                image = ImageTk.PhotoImage(pilimage)
                canvas.create_image(0, 0, image=image, anchor='nw')
                canvas.create_text(endX, startY + 10, font='DejavuSansLight 24', fill=color_txt, text=label)
                canvas.create_text(170, 30, font='DejavuSansLight 32', fill=color_txt_decision, text=label_decision)
                canvas.update()
                cs_canvas.delete('all')
                cs_canvas.create_image(0, 0, image=image, anchor='nw')
                cs_canvas.create_text(endX, startY + 10, font='DejavuSansLight 24', fill=color_txt, text=label)
                cs_canvas.create_text(340, 30, font='DejavuSansLight 32', fill=color_txt, text=label_visitor_info)
                if confirm:
                    if delay < 250:
                        cs_canvas.create_text(340, 30, font='DejavuSansLight 32', fill='#00FF00', text='Проходите')
                        lbl_info.configure(text='Подтверждено')
                    else:
                        cs_canvas.create_text(340, 30, font='DejavuSansLight 32', fill='#00FF00', text='')
                        lbl_info.configure(text='Определение личности')
                        confirm = False
                cs_canvas.update()
            else:
                frame = imutils.resize(frame, width=824)
                cv2.imwrite('img.jpg', frame)
                pilimage = Image.open('img.jpg')
                image = ImageTk.PhotoImage(pilimage)
                canvas.create_image(0, 0, image=image, anchor='nw')
                canvas.update()
                cs_canvas.create_image(0, 0, image=image, anchor='nw')
                cs_canvas.update()

            if os.path.isfile('img.jpg'): os.remove('img.jpg')
            if not isDetecting:
                # отключение определения личности
                vs.release()
                cv2.destroyAllWindows()
                btn_toDB.configure(state='disabled')
                return
            end = datetime.datetime.now()
            if confirm:
                if (end.microsecond - start.microsecond)/10000 > 0:
                    delay += (end.microsecond - start.microsecond)/10000
            else: delay = 0
    else:
        # если нажата кнопка во время работы определения личности
        isDetecting = False
        canvas.delete("all")
        btn_start.configure(text='Определить личность')
        lbl_info.configure(text='')
        root.update()

def newPerson():
    # окно добавления новой личности
    global win, txt_surname, txt_name, txt_secname, cb_post
    win = Toplevel()
    win.title("Новая личность")
    win.iconbitmap('icon.ico')
    win.resizable(width=False, height=False)
    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    w, h = 384, 210
    x = (sw - w) / 2
    y = (sh - h) / 2
    win.geometry('%dx%d+%d+%d' % (w, h, x, y))

    l1 = Frame(win)
    l1.pack(fill=BOTH, side='top')
    Label(l1, text='Фамилия:', font="DejavuSansLight 14").pack(side='left', padx=10, pady=5)
    txt_surname = Entry(l1, font="DejavuSansLight 12", width=23)
    txt_surname.pack(side='right', padx=(0,10), pady=10)

    l2 = Frame(win)
    l2.pack(fill=BOTH, side='top')
    Label(l2, text='Имя:', font="DejavuSansLight 14").pack(side='left', padx=10, pady=5)
    txt_name = Entry(l2,font="DejavuSansLight 12", width=23)
    txt_name.pack(side='right', padx=(0,10), pady=(0,10))

    l3 = Frame(win)
    l3.pack(fill=BOTH, side='top')
    Label(l3, text='Отчество:', font="DejavuSansLight 14").pack(side='left', padx=10, pady=5)
    txt_secname = Entry(l3,font="DejavuSansLight 12", width=23)
    txt_secname.pack(side='right', padx=(0,10), pady=(0,10))

    l4 = Frame(win)
    l4.pack(fill=BOTH, side='top')
    Label(l4, text='Должность:', font="DejavuSansLight 14").pack(side='left', padx=10, pady=5)
    cb_post = ttk.Combobox(l4, values=('Преподаватель', 'Студент'), font="DejavuSansLight 12", width=21)
    cb_post.current(0)
    cb_post.pack(side='right', padx=10, pady=(0,10))

    bf = Frame(win)
    bf.pack(fill=BOTH, side='bottom')
    btn_cancel = Button(bf, width="9", text="Отмена", pady="5", font="12", command=win.destroy)
    btn_cancel.pack(side='right', padx=(0, 10), pady="10")
    btn_ok = Button(bf, width="9", text="Ок", pady="5", font="12", command=addNewPerson)
    btn_ok.pack(side='right', padx=(0, 5), pady="10")

    if makemodal:
        win.focus_set()
        win.grab_set()
        win.wait_window()

def addNewPerson():
    # добавление новой личности в базу данных и переобучение нейронки
    global names, win, txt_surname, txt_name, txt_secname, cb_post, recognizer, cascadePath, faceCascade,lbl_info
    if len(txt_surname.get()) > 0 and len(txt_name.get()) > 0 and len(txt_secname.get()) > 0:
        try:
            # добавление записи в базу
            lbl_info.configure(text='Добавление личности')
            root.update()
            con_string = r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=.\FacesDB.mdb;'
            conn = db.connect(con_string)
            cur = conn.cursor()
            newRecord = (txt_surname.get(), txt_name.get(), txt_secname.get(), cb_post.get())
            sql = 'INSERT INTO Faces (face_surname, face_name, face_secname, face_post) VALUES ' + str(newRecord)
            cur.execute(sql)
            conn.commit()
            print('Data Inserted')
            sql = 'SELECT * FROM Faces'
            cur.execute(sql)
            row = cur.fetchall()[len(cur.fetchall())-1]
            names.append((row[0], row[1], row[2], row[3], row[4]))
            win.destroy()

            # получение образцов лица
            lbl_info.configure(text='Смотрите в камеру')
            root.update()
            cam = cv2.VideoCapture(0)
            cam.set(3, 640)
            cam.set(4, 480)
            face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Детектор лица
            face_id = int(row[0])
            print("\n [INFO] Initializing face capture. Look the camera and wait ...")
            count = 0
            while (True):
                ret, frame = cam.read()
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    count += 1
                    cv2.imwrite("Database/." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])
                    frame = imutils.resize(frame, width=824)
                    cv2.imwrite('img.jpg', frame)
                    pilimage = Image.open('img.jpg')
                    image = ImageTk.PhotoImage(pilimage)
                    canvas.create_image(0, 0, image=image, anchor='nw')
                    canvas.update()
                    cs_canvas.create_image(0, 0, image=image, anchor='nw')
                    cs_canvas.update()
                if count >= 100:
                   break
            print("\n [INFO] Exiting Program and cleanup stuff")
            cam.release()
            cv2.destroyAllWindows()
            canvas.delete("all")
            canvas.update()
            cs_canvas.delete("all")
            cs_canvas.update()
            if os.path.isfile('img.jpg'): os.remove('img.jpg')

            # обучение нейросети
            lbl_info.configure(text='Обучение')
            path = 'Database'
            _recognizer = cv2.face.LBPHFaceRecognizer_create()
            detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
            print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
            faces, ids = getImagesAndLabels(path, detector)
            _recognizer.train(faces, np.array(ids))
            _recognizer.write('trainer/trainer.yml')
            print("\n [INFO] {0} faces trained.".format(len(np.unique(ids))))

            # Перезапуск нейросети
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read('trainer/trainer.yml')
            faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        except db.Error as e:
            print("Error in Connection", e)
    lbl_info.configure(text='Личность добавлена')
    root.update()

# получение образцов лица
def getImagesAndLabels(path, detector):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)
    return faceSamples, ids

def delPerson():
    # Окно удаления личности
    global names, win, cb_face
    win = Toplevel()
    win.title("Удалить личность")
    win.iconbitmap('icon.ico')
    win.resizable(width=False, height=False)
    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    w, h = 384, 106
    x = (sw - w) / 2
    y = (sh - h) / 2
    win.geometry('%dx%d+%d+%d' % (w, h, x, y))

    l1 = Frame(win)
    l1.pack(fill=BOTH, side='top')
    lbl_face = Label(l1, text='Личность:', font="DejavuSansLight 14")
    lbl_face.pack(side='left', padx=10, pady=5)
    values = []
    for row in names:
        values.append(row[1]+' '+row[2][0]+'.'+row[3][0]+'. '+row[4])
    cb_face = ttk.Combobox(l1, values=values, font="DejavuSansLight 12", width=25)
    cb_face.current(0)
    cb_face.pack(side='left', padx=10, pady=10)

    bf = Frame(win)
    bf.pack(fill=BOTH, side='bottom')
    btn_cancel = Button(bf, width="9", text="Отмена", pady="5", font="12", command=win.destroy)
    btn_cancel.pack(side='right', padx=(0, 10), pady="10")
    btn_ok = Button(bf, width="9", text="Ок", pady="5", font="12", command=confirmDeletePerson)
    btn_ok.pack(side='right', padx=(0, 5), pady="10")

    if makemodal:
        win.focus_set()
        win.grab_set()
        win.wait_window()

def confirmDeletePerson():
    # Подтверждение удаления личности
    global names, win, cb_face, recognizer, faceCascade, lbl_info
    face_id = names[cb_face.current()][0]
    try:
        # Удаление записи из базы данных
        lbl_info.configure(text='Удаление личности')
        root.update()
        con_string = r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=.\FacesDB.mdb;'
        conn = db.connect(con_string)
        cur = conn.cursor()
        sql = 'DELETE FROM Faces WHERE face_id='+str(face_id)
        cur.execute(sql)
        conn.commit()
        print('Data Deleted')
        win.destroy()

        # Удаление образцов лица
        path = 'Database'
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        for imagePath in imagePaths:
            id = int(os.path.split(imagePath)[1].split(".")[1])
            if id == face_id:
                os.remove(imagePath)

        # Переобучение нейросети
        lbl_info.configure(text='Обучение')
        _recognizer = cv2.face.LBPHFaceRecognizer_create()
        detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
        print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
        faces, ids = getImagesAndLabels(path, detector)
        _recognizer.train(faces, np.array(ids))
        _recognizer.write('trainer/trainer.yml')
        print("\n [INFO] {0} faces trained.".format(len(np.unique(ids))))

        # Перезапуск нейросети
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('trainer/trainer.yml')
        faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    except db.Error as e:
        print("Error in Connection", e)
    lbl_info.configure(text='Личность удалена')
    root.update()

def toDB():
    # Окно подтверждения личности
    global win, current_face, cb_facility, txt_room, current_date, current_time
    win = Toplevel()
    win.title("Подтвердить личность")
    win.iconbitmap('icon.ico')
    win.resizable(width=False, height=False)
    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    w, h = 384, 176
    x = (sw - w) / 2
    y = (sh - h) / 2
    win.geometry('%dx%d+%d+%d' % (w, h, x, y))

    l1 = Frame(win)
    l1.pack(fill=BOTH, side='top')
    face_name = current_face[1]+' '+current_face[2]+' '+current_face[3]
    lbl_face = Label(l1, text=face_name, font="DejavuSansLight 18")
    lbl_face.pack(side='left', padx=10, pady=5)

    l2 = Frame(win)
    l2.pack(fill=BOTH, side='top')
    lbl_post = Label(l2, text=current_face[4], font="DejavuSansLight 14")
    lbl_post.pack(side='left', padx=10, pady=(0,10))

    l3 = Frame(win)
    l3.pack(fill=BOTH, side='top')
    now = datetime.datetime.now()
    current_datetime = now.strftime('%d.%m.%Y %H:%M')
    current_date = now.strftime('%d.%m.%Y')
    current_time = now.strftime('%H:%M')
    lbl_datetime = Label(l3, text=current_datetime, font="DejavuSansLight 14")
    lbl_datetime.pack(side='left', padx=10, pady=(0,10))

    bf = Frame(win)
    bf.pack(fill=BOTH, side='bottom')
    btn_cancel = Button(bf, width="9", text="Отмена", pady="5", font="12", command=win.destroy)
    btn_cancel.pack(side='right', padx=(0, 10), pady="10")
    btn_ok = Button(bf, width="9", text="Ок", pady="5", font="12",command=confirmPerson)
    btn_ok.pack(side='right', padx=(0, 5), pady="10")

    if makemodal:
        win.focus_set()
        win.grab_set()
        win.wait_window()

def confirmPerson():
    # добавление новой записи подтверждения личности в базу данных
    global win, current_face, cb_facility, current_date, current_time, lbl_info, cb_facility, confirm
    try:
        con_string = r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=.\FacesDB.mdb;'
        conn = db.connect(con_string)
        cur = conn.cursor()
        newRecord = (str(current_face[0]), current_date, current_time, cb_facility.get())
        sql = 'INSERT INTO Log (face_id, arrival_date, arrival_time, facility) VALUES ' + str(newRecord)
        cur.execute(sql)
        conn.commit()
        print('Data Inserted')
    except db.Error as e:
        print("Error in Connection", e)
    win.destroy()
    confirm = True
    lbl_info.configure(text='Подтверждено')
    root.update()

def openLog():
    # Окно журнала
    global win, table, table_data, txt_surname, txt_name, txt_secname, txt_date, txt_time, cb_post, cb_facility_
    win = Toplevel()
    win.title("Журнал")
    win.iconbitmap('icon.ico')
    win.resizable(width=False, height=False)
    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    w, h = 768, 640
    x = (sw - w) / 2
    y = (sh - h) / 2
    win.geometry('%dx%d+%d+%d' % (w, h, x, y))

    l1 = Frame(win, bg=color_bg)
    l1.pack(fill=BOTH, side='top')
    l2 = Frame(win, bg=color_bg)
    l2.pack(fill=BOTH, side='top')
    l3 = Frame(win, bg=color_bg)
    l3.pack(fill=BOTH, side='top')
    l4 = Frame(win, bg=color_bg)
    l4.pack(fill=BOTH, side='top')
    Label(l1, text='Фамилия', font="DejavuSansLight 12", width=10, bg=color_bg).pack(side='left', padx=5, pady=5)
    Label(l2, text='Имя', font="DejavuSansLight 12", width=10, bg=color_bg).pack(side='left', padx=5, pady=5)
    Label(l3, text='Отчество', font="DejavuSansLight 12", width=10, bg=color_bg).pack(side='left', padx=5, pady=5)

    txt_surname = Entry(l1, font="DejavuSansLight 12", width=10)
    txt_surname.pack(side='left', padx=5, pady=5)
    txt_name = Entry(l2, font="DejavuSansLight 12", width=10)
    txt_name.pack(side='left', padx=5, pady=5)
    txt_secname = Entry(l3, font="DejavuSansLight 12", width=10)
    txt_secname.pack(side='left', padx=5, pady=5)

    Label(l1, text='Должность', font="DejavuSansLight 12", width=10, bg=color_bg).pack(side='left', padx=5, pady=5)
    Label(l2, text='Корпус', font="DejavuSansLight 12", width=10, bg=color_bg).pack(side='left', padx=5, pady=5)

    cb_post = ttk.Combobox(l1, values=('Любой', 'Студент', 'Преподаватель'), font="DejavuSansLight 12", width=10)
    cb_post.current(0)
    cb_post.pack(side='left', padx=5, pady=5)
    cb_facility_ = ttk.Combobox(l2, values=('Любой', 'A', 'B', 'C', 'D'), font="DejavuSansLight 12", width=10)
    cb_facility_.current(0)
    cb_facility_.pack(side='left', padx=5, pady=5)

    Label(l1, text='Дата', font="DejavuSansLight 12", width=10, bg=color_bg).pack(side='left', padx=5, pady=5)
    Label(l2, text='Время', font="DejavuSansLight 12", width=10, bg=color_bg).pack(side='left', padx=5, pady=5)

    txt_date = Entry(l1, font="DejavuSansLight 12", width=10)
    txt_date.pack(side='left', padx=5, pady=5)
    txt_time = Entry(l2, font="DejavuSansLight 12", width=10)
    txt_time.pack(side='left', padx=5, pady=5)

    lt = Frame(win, bg=color_bg)
    lt.pack(expand=1, fill=BOTH, side='top')

    # Таблица
    heads = ['Фамилия', 'Имя', 'Отчество', 'Должность', 'Корпус', 'Дата прибытия', 'Время прибытия']
    table = ttk.Treeview(lt, show='headings')
    table['columns'] = heads

    scroll_pane = ttk.Scrollbar(lt, command=table.yview)
    table.configure(yscrollcommand=scroll_pane.set)
    scroll_pane.pack(side='right', fill='y')
    table.pack(expand='yes', fill='both')
    for header in heads:
        table.heading(header, text=header, anchor='center')
        table.column(header, anchor='center', width=int(w/8))
    createTable()
    bf = Frame(win, bg=color_bg)
    bf.pack(fill=BOTH, side='bottom')

    btn_cancel = Button(bf, width="9", text="Закрыть", pady="5", font="12", command=win.destroy, bg=color_buttons)
    btn_cancel.pack(side='right', padx=(0, 10), pady="10")
    btn_OK = Button(l4, width="9", text="Поиск", pady="5", font="12", command=createTable, bg=color_buttons)
    btn_OK.pack(side='right', padx=(0, 10), pady="10")

    if makemodal:
        win.focus_set()
        win.grab_set()
        win.wait_window()

def createTable():
    global table, table_data, txt_surname, txt_name, txt_secname, txt_date, txt_time, cb_post, cb_facility_
    table.delete(*table.get_children())
    # Запрос в базу
    table_data = []
    try:
        con_string = r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=.\FacesDB.mdb;'
        conn = db.connect(con_string)
        cur = conn.cursor()
        post, facility = '', ''
        if cb_post.current() != 0: post = cb_post.get()
        if cb_facility_.current() != 0: facility = cb_facility_.get()
        sql = 'SELECT face_surname, face_name, face_secname, ' \
              'face_post, facility, arrival_date, arrival_time ' \
              'FROM Log INNER JOIN Faces ON Log.face_id = Faces.face_id ' \
              'WHERE face_surname LIKE \'%'+txt_surname.get()+'%\' ' \
              'AND face_name LIKE \'%'+txt_name.get()+'%\' ' \
              'AND face_secname LIKE \'%'+txt_secname.get()+'%\' ' \
              'AND face_post LIKE \'%'+post+'%\' ' \
              'AND facility LIKE \'%'+facility+'%\' ' \
              'AND arrival_date LIKE \'%'+txt_date.get()+'%\' ' \
              'AND arrival_time LIKE \'%'+txt_time.get()+'%\''
        cur.execute(sql)
        for row in cur.fetchall():
            _ = str(row[5]).split()[0].split('-')
            date = _[2]+'.'+_[1]+'.'+_[0]
            _ = str(row[6]).split()[1].split(':')
            time = _[0]+':'+_[1]
            table_data.append((str(row[0]),str(row[1]),str(row[2]), str(row[3]), str(row[4]),
                         date, time))
    except db.Error as e:
        print("Error in Connection", e)
    for row in table_data:
        table.insert('', 'end', values=row)

def openManual():
    # открыть руководство пользователя
    # вроде как это работает только на винде
    os.startfile('manual.pdf')

def onClosing():
    global isDetecting, lbl_info
    if isDetecting:
        lbl_info.configure(text='Завершите все процессы')
    else:
        if os.path.isfile('img.jpg'): os.remove('img.jpg')
        sys.exit(1)

# admin screen setup
screen_width = 1024
screen_height = 640
root = Tk()
root.title("Система контроля доступа")
root.iconbitmap('icon.ico')
x = (root.winfo_screenwidth() - screen_width) / 2
y = (root.winfo_screenheight() - screen_height) / 2
root.geometry('%dx%d+%d+%d' % (screen_width, screen_height, x, y))
root.resizable(width=False, height=False)
root.protocol("WM_DELETE_WINDOW", onClosing)

# visitor screen setup
csreen = Toplevel()
csreen.title("Система контроля доступа")
csreen.iconbitmap('icon.ico')
csreen.resizable(width=False, height=False)
csreen.geometry('748x620')
csreen.protocol("WM_DELETE_WINDOW", onClosing)

if makemodal:
    win.focus_set()
    win.grab_set()
    win.wait_window()

# colors
color_bg = '#E4E6DD'
color_buttons = '#E4E6DD'
color_canvas = '#FFFFFF'
color_canvas_highlight = '#A0A1A0'

# frames
lf = Frame(root, width='256', bg=color_bg)
lf.pack(side="left", fill="y")
rf = Frame(root, width='768', bg=color_bg)
rf.pack(side="left", fill="y")
bf = Frame(lf, bg=color_bg)
bf.pack(side="bottom", fill="x")

# buttons
btn_start = Button(lf, text="Определить личность", width=33, height=2, command=detectPerson, bg=color_buttons)
btn_start.pack(side="top", padx=10, pady=10)

btn_toDB = Button(lf, text="Пропустить", width=33, height=2, command=toDB, bg=color_buttons, state='disabled')
btn_toDB.pack(side="top", padx=10, pady=(0,10))

btn_log = Button(lf, text="Открыть журнал", width=33, height=2, command=openLog, bg=color_buttons)
btn_log.pack(side="top", padx=10, pady=(0,10))

btn_new_person = Button(lf, text="Новая личность", width=33, height=2, command=newPerson, bg=color_buttons)
btn_new_person.pack(side="top", padx=10, pady=(0,10))

btn_new_person = Button(lf, text="Удалить личность", width=33, height=2, command=delPerson, bg=color_buttons)
btn_new_person.pack(side="top", padx=10, pady=(0,10))

btn_manual = Button(bf, text="?", width=5, height=2, command=openManual, bg=color_buttons)
btn_manual.pack(side="left", padx=10, pady=(0,10))

# info
lf_cb = Frame(lf, width='256', bg=color_bg)
lf_cb.pack(side="top", fill="y")
lbl_facility = Label(lf_cb, text="Корпус:", font="DejavuSansLight 14", bg=color_bg)
lbl_facility.pack(side='left', padx=10, pady=(0, 10))
cb_facility = ttk.Combobox(lf_cb, values=('A', 'B', 'C', 'D'), font="DejavuSansLight 12", width=7)
cb_facility.current(0)
cb_facility.pack(side='left', padx=10, pady=(0, 10))

lbl_info = Label(lf, text='', font="DejavuSansLight 14", bg=color_bg)
lbl_info.pack(side="top", padx=10, pady=(0,10))

# canvas
canvas = Canvas(rf, width=748, height=620, highlightthickness=2,highlightbackground=color_canvas_highlight, bg=color_canvas)
canvas.pack(side="top", padx=(0,10), pady=10)

cs_canvas = Canvas(csreen, width=748, height=620, highlightthickness=2,highlightbackground=color_canvas_highlight, bg=color_canvas)
cs_canvas.pack(side="top", padx=10, pady=10)

root.mainloop()