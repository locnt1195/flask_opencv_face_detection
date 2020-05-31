from flask import Flask, request, Response, render_template
import cv2
import numpy as np
import time
app = Flask(__name__)
IMAGE = 'images/face.jpg'
IMAGE_FACE_DETECTION = 'images/face_detection.jpg'


@app.route('/')
def home():
    return render_template('index.html')


# Task 1
@app.route("/hello1")
def hello1():
    args = request.args
    if args:
        return args
    else:
        return "Hello"


# Task 2
@app.route("/hello2")
def hello2():
    args = request.args
    name = args.get('name', '')
    return "Hello %s" % name


# Task 3
def gen():
    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()

    if not ret:
        print("Error: failed to capture image")
        return False

    cv2.imwrite(IMAGE, frame)
    return (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + open(IMAGE, 'rb').read() + b'\r\n')


# Task 4
@app.route('/grabimage')
def grabimage():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# Task 4
def face_detection():
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    img = cv2.imread(IMAGE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

    cv2.imwrite(IMAGE_FACE_DETECTION, img)
    return (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + open(IMAGE_FACE_DETECTION, 'rb').read() + b'\r\n')

# Task 6
@app.route('/facedetection')
def facedetection():
    return Response(face_detection(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8888)
