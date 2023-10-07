import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io, feature
from scipy.stats import entropy
from itertools import chain
from skimage.feature import graycomatrix, graycoprops
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from skimage.util import img_as_float
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_score
from scipy.spatial import distance
from operator import add
from skimage import color
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries


def GLCM(m,distancia,A,scale):
  '''
  m: matriz inicial
  distancia: distancia de movimiento 
  A: ángulo
  Scale: tamaño de la matriz de co-ocurrencia
  '''
  glcm = graycomatrix(m, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

  glcm = glcm[:, :, 0, 0]
  var = glcm.var()
  mean = glcm.mean()
  std = glcm.std()
  glcm_h = glcm.reshape((-1,1))
  h = entropy(glcm_h, base=2)
  return [var, std, mean]


def vectorizar(glcm):
  var = glcm.var()
  mean = glcm.mean()
  std = glcm.std()

  glcm_h = glcm.reshape((-1,1))

  return [mean, var, std]


# Crear ventanas en cada imagen
# Cada ventana se guarda en una lista 
def generar_ventanas(imagen, distancia, tamaño, ventanaTam):
  n = distancia*2
#   n = ventanaTam
  ventanas = list()
  for i in range(0,tamaño-n,distancia):
    for j in range(0,tamaño-n,distancia):
      ventanas.append(imagen[i:i+n,j:j+n])
      
  return ventanas


# Etiquetar cada ventana con la imagen que corresponden
def etiquetar(lista_muestras, etiqueta):
  return [(x, etiqueta) for x in lista_muestras]

def sacar_glcm_entrenamiento(url, distancia, angulo, etiqueta):
  # cargamos la imágen
  img = io.imread(url)
  data1 = img[:,:,0]
  data = img[:,:,0]
#   data = data[160, 160]

  # establecemos el valor posible de pixeles
  escala = 256

  # Se generan las ventanas
  lista_ventanas = generar_ventanas(data, distancia, data1.shape[0], 16)

  # Generamos las glcm de cada prueba y calculamos los vectores de las ventanas
    # Estos los usamos para entrenar el modelo 
  lista_ventanas_vectores = [GLCM(ventana, distancia, angulo, escala) for ventana in lista_ventanas]

  prom = [0,0,0]
  for f in range(len(lista_ventanas_vectores)):
        prom = list(map(add, lista_ventanas_vectores[f], prom))
  vector_textura = [x / len(lista_ventanas_vectores) for x in prom]

  return vector_textura


#Cambié la distancia a 16
train_set_1 = sacar_glcm_entrenamiento("./imagenes/D65.bmp",80,0,"1")
train_set_2 = sacar_glcm_entrenamiento("./imagenes/D64.bmp",80,0,"2")
train_set_3 = sacar_glcm_entrenamiento("./imagenes/D49.bmp",80,0,"3")
train_set_4 = sacar_glcm_entrenamiento("./imagenes/D101.bmp",80,0,"4")
train_set_5 = sacar_glcm_entrenamiento("./imagenes/D6.bmp",80,0,"5")
train_set_6 = sacar_glcm_entrenamiento("./imagenes/D16.bmp",80,0,"6")



# fig = plt.figure(figsize=(80, 10))

# ax = fig.add_subplot(3,4, 1)
# img = io.imread("./imagenes/D65.bmp")
# ax.set_title('D65')
# ax.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=255)
# ax = fig.add_subplot(3,4, 2)
# img = io.imread("./imagenes/D64.bmp")
# ax.set_title('D64')
# ax.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=255)
# ax = fig.add_subplot(3,4, 3)
# img = io.imread("./imagenes/D49.bmp")
# ax.set_title('D49')
# ax.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=255)
# ax = fig.add_subplot(3,4, 4)
# img = io.imread("./imagenes/D101.bmp")
# ax.set_title('D101')
# ax.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=255)
# ax = fig.add_subplot(3,4, 5)
# img = io.imread("./imagenes/D6.bmp")
# ax.set_title('D6')
# ax.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=255)
# ax = fig.add_subplot(3,4, 6)
# img = io.imread("./imagenes/D16.bmp")
# ax.set_title('D16')
# ax.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=255)

# plt.show()


print(train_set_1)

def distancia(vector1, textura):
    correlation_distance = distance.correlation(vector1, textura)
    euclidean_distance = distance.euclidean(vector1, textura)
    hamming_distance = distance.hamming(vector1, textura)
    manhattan_distance = distance.cityblock(vector1, textura)
    return manhattan_distance

def generar_prediccion(imagen, distancia2, tamaño1, tamaño2,angulo,escala):

  n = distancia2*2
  distancias = list()
  
  nuevaimagen = np.zeros_like(imagen)
  for i in range(0,tamaño1,n):
    for j in range(0,tamaño2,n):
      print(i)
      print(j)
      Vector_Prediccion=GLCM((imagen[i:i+n,j:j+n]),distancia2,angulo,escala)
      distancias.append(distancia(Vector_Prediccion, train_set_1))
      distancias.append(distancia(Vector_Prediccion, train_set_2))
      distancias.append(distancia(Vector_Prediccion, train_set_3))
      distancias.append(distancia(Vector_Prediccion, train_set_4))

      print(distancias)

      min = np.amin(distancias)
      print(min)
      index = distancias.index(min)
      print(index)
      if index == 0:
        nuevaimagen[i:i+n,j:j+n]=30
      elif index == 1:
        nuevaimagen[i:i+n,j:j+n]=64
      elif index == 2:
        nuevaimagen[i:i+n,j:j+n]=80
      elif index == 3:
        nuevaimagen[i:i+n,j:j+n]=170
      elif index == 4:
        nuevaimagen[i:i+n,j:j+n]=800
      else :
        nuevaimagen[i:i+n,j:j+n]=255

      distancias.clear()
  return nuevaimagen

# data = io.imread("./imagenes/imgCompuesta1.png")
data = cv2.imread("./imagenes/D96.bmp",0)
escala = 256
img1n = generar_prediccion(data, 80, data.shape[0],data.shape[1] ,0,escala)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(1,2, 1)
ax.imshow(data, cmap=plt.cm.gray, vmin=0, vmax=255)
print(img1n)
ax = fig.add_subplot(1,2, 2)
ax.imshow(img1n, cmap=plt.cm.gray, vmin=0, vmax=255)
print(img1n)

plt.show()

# data = io.imread("./imagenes/imgCompuesta2.png")
data = cv2.imread("./imagenes/Composicion.png", 0)
escala = 256
img2n = generar_prediccion(data, 80, data.shape[0] , data.shape[1] ,0,escala)
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(1,2, 1)
ax.imshow(data, cmap=plt.cm.gray, vmin=0, vmax=255)
# print(img1n)
ax = fig.add_subplot(1,2, 2)
ax.imshow(img2n, cmap=plt.cm.gray, vmin=0, vmax=255)
print(img1n)

plt.show()

image = img_as_float(io.imread("./imagenes/Composicion.png"))
# image = img_as_float(io.imread("./imagenes/imgCompuesta1.png"))
for numSegments in (100, 200, 300, 50):
	segments = slic(image, n_segments = numSegments, sigma = 5)
	fig = plt.figure("Superpixels -- %d segments" % (numSegments))
	ax = fig.add_subplot(1, 1, 1)
	ax.imshow(mark_boundaries(image, segments))    
plt.show()

# segments1 = slic(image, n_segments = 50, sigma = 5)
# for s_pixel in segments:
#         for i in range(segments.shape[0]):
#                 for j in range(segments.shape[1]):
#                    print(segments[i,j])