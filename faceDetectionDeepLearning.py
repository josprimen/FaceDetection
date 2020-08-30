import mtcnn
from mtcnn.mtcnn import MTCNN
#print(mtcnn.__version__)
from matplotlib import pyplot
from matplotlib.patches import Rectangle

from PIL import Image




def draw_image_with_boxes(filename, result_list):
    #cargamos la imagen
    data = pyplot.imread(filename)
    #imprimimos la imagen
    pyplot.imshow(data)
    #obtenemos el contexto para dibujar los cuadros delimitadores
    ax = pyplot.gca()
    #imprimimos cada cuadro delimitador de la solucion
    for result in result_list:
        #obtener las coordenadas
        x, y, width, height = result['box']
        #crear los cuadros
        rect = Rectangle((x,y), width, height, fill=False, color='white')
        #dibujamos los cuadros
        ax.add_patch(rect)
    #mostramos por pantalla
    pyplot.show()



filename = 'image/foto grupal.jpg'
#Usamos la funcion imread de pyplot en lugar de la de openv
pixels = pyplot.imread(filename)
#cargamos el modelo pre-entrenado que nos proporciona la libreria
detector = MTCNN()
#Detectamos los rostros en la imagen usando el modelo
faces = detector.detect_faces(pixels)
for face in faces:
    print(face)
#usamos una función para mostrar todos los datos por pantalla, tenemos que pasar la imagen original y las caras detectadas
draw_image_with_boxes(filename, faces)


