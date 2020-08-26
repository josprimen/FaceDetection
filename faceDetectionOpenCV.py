import cv2
from cv2 import CascadeClassifier
from cv2 import imread
#print(cv2.__version__)

#Cargamos un modelo pre-entrenado para reconocimiento facial y creamos un modelo en cascada con el
classifier = CascadeClassifier('haarcascade_frontalface_default.xml')

#cargamos una foto de prueba
pixels = imread('test1.jpg')

#realizamos el reconocimiento facial
bboxes = classifier.detectMultiScale(pixels)
#representamos los cuadros delimitadores de los rostros detectados
for box in bboxes:
    print(box)
