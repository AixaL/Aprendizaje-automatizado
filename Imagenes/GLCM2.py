import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import cv2
from skimage.feature import graycomatrix, graycoprops
from operator import add
from scipy.spatial import distance
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

minimos = []

def sliding_window(image, stepSize, windowSize):
  windows = []
  for y in range(0, image.shape[0], stepSize):
    for x in range(0, image.shape[1], stepSize):
        windows.append(image[y:y + windowSize[1], x:x + windowSize[0]])
  return windows


def sliding_window_put(image, matriz, valorInt, vector_textura, stepSize, windowSize):
  distancias = [] 
  windows = []
  for y in range(0, image.shape[0], stepSize):
    for x in range(0, image.shape[1], stepSize):
        window = image[y:y + windowSize[1], x:x + windowSize[0]]
        wind = (x,y)
        windows.append(wind)
        vector_window = extractFeatures(window)
#        print(vector_window)
#        print(vector_textura)
        dis = distancia(vector_window, vector_textura)
#        print(dis)
        distancias.append(dis)
#        a = input("Continuar:\n")
        if dis == min((distancias)):
            minimo = dis
#  print(minimo)
#  print("El minimo fue el ultimo numero")
  indice = distancias.index(minimo)
  wx, wy = windows[indice]
  matriz[wy:wy + windowSize[1], wx:wx + windowSize[0]] = valorInt
  minimos.append(indice)       


def distancia(vector1, vector2):
#    hamming_distance = distance.hamming(list(vector1), list(vector2))
    manhattan_distance = distance.cityblock(vector1, vector2)
    return manhattan_distance

def extractFeatures(window):
  features = graycomatrix(window, [1], [0])
  features = features[:, :, 0, 0]
  var = features.var()
#  mean = features.mean()
  std = features.std()
  glcm_h = features.reshape((-1,1))
  h = entropy(glcm_h, base=2)
  return [var, std, h[0]]


def vectordetextura(image, stepSize, windowSize):
    features = []
    prom = [0,0,0]
    windows = sliding_window(image, stepSize, windowSize)
    for window in windows:
        featureVector = extractFeatures(window)
        features.append(featureVector)
    for f in range(len(features)):
        prom = list(map(add, features[f], prom))
    vector_textura = [x / len(features) for x in prom]
    return vector_textura

def superpixeles(img):
    vectores = []
    img_slic = slic(image, n_segments = 300, sigma = 5, channel_axis=None)
    # print(img_slic.shape)
    #glcm = graycomatrix(img_slic, [1], [0])  # Calculate the GLCM "one pixel to the right"
    #filt_glcm = glcm[1:, 1:, :, :]           # Filter out the first row and column
    #greycoprops(filt_glcm, prop='contrast')
    plt.figure()
    #plt.imshow(filt_glcm)
    filtra = img_slic
    #pixel = 1
    for pixel in range(0,254):
        for x in range(320):
            for i in range(len(img_slic[x][:])):
                if img_slic[x][i] > pixel or img_slic[x][i] < pixel:
                    filtra[x][i] = 0
                    imagen = img * filtra
                    if pixel  == 1:
                        plt.figure()
                        plt.imshow(filtra)
                    if pixel > 0:
                        vector = extractFeatures(filtra)
                        vectores.append(vector)
    return vectores
    #plt.imshow(mark_boundaries(img, img_slic)[:][:])

"Main:"

stepSize = 160
windowSize = (160, 160)

#image = cv2.imread('ComposicionJ.png', 0) 

image = cv2.imread('imgCompuesta2.png', 0) 
imagen1 = cv2.imread('D101.bmp', 0)
imagen2 = cv2.imread('D49.bmp', 0)
imagen3 = cv2.imread('D64.bmp', 0)
imagen4 = cv2.imread('D65.bmp', 0)
# imagen4 = cv2.resize(cv2.imread('Piedras.jpg', 0), (640,640))

#image = cv2.imread('imgCompuesta1.png', 0) 
#imagen1 = cv2.imread('D6.bmp', 0)
#imagen2 = cv2.imread('D64.bmp', 0)
#imagen3 = cv2.imread('D49.bmp', 0)
#imagen4 = cv2.imread('D101.bmp', 0)

vector_textura1 = vectordetextura(imagen1, stepSize, windowSize)
# print(vector_textura1)
vector_textura2 = vectordetextura(imagen2, stepSize, windowSize)
vector_textura3 = vectordetextura(imagen3, stepSize, windowSize)
vector_textura4 = vectordetextura(imagen4, stepSize, windowSize)

imagen_ceros = np.zeros((image.shape[0], image.shape[1]))

sliding_window_put(image, imagen_ceros, 0, vector_textura1, stepSize, windowSize)
sliding_window_put(image, imagen_ceros, 60, vector_textura2, stepSize, windowSize)
sliding_window_put(image, imagen_ceros, 150, vector_textura3, stepSize, windowSize)
sliding_window_put(image, imagen_ceros, 256, vector_textura4, stepSize, windowSize)

plt.figure()
plt.imshow(image, cmap = "gray")

plt.figure()
plt.imshow(imagen_ceros, cmap = "gray")


plt.show()

#v = superpixeles(image)