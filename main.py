from shared import *

img = cv.imread('data/img/camera.png')
gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
classifier = cv.CascadeClassifier(
    cv.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
face = classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=10, minSize=(130, 130)
)

print(len(face))

