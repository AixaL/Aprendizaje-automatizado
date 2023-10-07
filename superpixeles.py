# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse

# load the image and convert it to a floating point data type
image = img_as_float(io.imread("./imagenes/Composicion.png"))
# image = img_as_float(io.imread("./imagenes/imgCompuesta1.png"))
for numSegments in (100, 200, 300, 50):
	segments = slic(image, n_segments = numSegments, sigma = 5)
	# segments = slic(image, n_segments = numSegments, sigma = 5, channel_axis=None)
	fig = plt.figure("Superpixels -- %d segments" % (numSegments))
	ax = fig.add_subplot(1, 1, 1)
	ax.imshow(mark_boundaries(image, segments))
	plt.axis("off")
        
  

plt.show()
