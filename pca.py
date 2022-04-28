from PIL import Image as pimg
import matplotlib.image as img
import numpy as np
from numpy.linalg import eig

# Number of images
n = 40

# All images are of same dimension
# Saving the dimension
image_dimensions = img.imread('images/f1.pgm').shape

# Taking all the images and converting them to 1d array
# Next, append all the array into one array
# Reshape to get each image in a row
# Take transpose of this array to get 'F' matrix
# Where each image is in column
image_matrix_list = []
for i in range(1,41):
    image_matrix_list = np.append(image_matrix_list,img.imread(f'images/f{i}.pgm').flatten())
reshaped_matrix = np.reshape(image_matrix_list,(n,image_dimensions[0]*image_dimensions[1]))
image_matrix = reshaped_matrix.T

# Subtract mean
mean = image_matrix.mean(axis=1,keepdims=True)/40
F = image_matrix - mean

# R = FxFT
R = np.dot(F,F.T)

# Find Eigen values and Eigen Vectors
evalue,evector = eig(R)

# set k values as 10% of the original
k = int(evector.shape[0]*0.1)
E = evector[:,:k]
P = np.dot(E.T,F)

reconstructed_image = np.dot(E,P)

image_index = 1 # The image number which is to be reconstructed

# Reconstructed Image
a = pimg.fromarray(np.reshape(reconstructed_image.real.T[image_index-1],(image_dimensions[0],image_dimensions[1])))
a.convert("L").save(f"output/result_{image_index}_{k}.jpg")

# Original Image
b = pimg.fromarray(img.imread(f'images/f{image_index}.pgm'))
b.convert("L").save(f"output/og_{image_index}_{k}.jpg")