import cv2
from cv2 import CascadeClassifier
from cv2 import imread
from cv2 import rectangle
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
#print(cv2.__version__)

#Cargamos un modelo pre-entrenado para reconocimiento facial y creamos un modelo en cascada con el
classifier = CascadeClassifier('haarcascade_frontalface_default.xml')

#cargamos una foto de prueba
pixels = imread('image/foto grupal.jpg')

#realizamos el reconocimiento facial. 1.1 significa ampliar la foto un 10% y 3 indica el nivel de fiabilidad de la deteccion
bboxes = classifier.detectMultiScale(pixels, 1.1, 3)
#representamos los cuadros delimitadores de los rostros detectados
for box in bboxes:
    print(box)
    # para mostrar la imagen original con los cuadros delimitadores dibujados encima haremos lo siguiente:
    # extraemos cada propiedad del resultado (x e y de la esquina inferior izq del cuadro delimitador y su altura y anchura)
    x, y, width, height = box
    x2, y2 = x + width, y + height
    # dibujamos un rectangulo sobre los pixels
    rectangle(pixels, (x, y), (x2, y2), (0, 0, 255), 1)

#imprimimos la imagen con la deteccion y la mantenamos abierta hasta que el usuario pulse una tecla
#mostramos la imagen
imshow('face detection', pixels)
#mantenemos la ventana abierta hasta que se pulse una tecla
waitKey(0)
#cerramos la ventana
destroyAllWindows()