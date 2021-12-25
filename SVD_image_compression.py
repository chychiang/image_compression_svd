
from skimage import data, img_as_float
from skimage.color import rgb2gray
from numpy.linalg import svd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image


def load_image(image_file):
	img = Image.open(image_file)
	return img


st.title("Image Compression With SVD")

# let user choose to upload to choose a stock image
img = None # init img
option = st.selectbox("Choose to upload or select a image", ('I\'ll upload my own image', 'Choose a random image for me', ))
if option == 'I\'ll upload my own image':
    # allow users to upload their own image
    img_file = st.file_uploader("Upload your own image")
    if img_file is not None:
        img = load_image(img_file)
else:
    img = data.camera() # convert image to grayscale
    

if img is not None: 
    img = rgb2gray(img_as_float(img)) # convert image to grayscale
    U, S, V = svd(img, full_matrices=False)
    
    # let users choose the k value
    st.write("Play with the # of vectors to reconstruct the image with")
    k = st.slider("Select the # of vectors", min_value=1, max_value=250, value=125, step=1)
    Sk = np.diag(S[:k])
    # print(U[:, :k].shape, Sk.shape, V[:, :k].shape)
    compressed_img = U[:, :k] @ Sk @ V[:k, :]

    # plot images in two columns
    col1, col2 = st.columns(2)

    col1.image(img, clamp=True, caption="Original Image")
    col2.image(compressed_img, clamp=True, caption="Compressed Image")

    original_size = img.size * img.itemsize
    compressed_size = S[:k].size * S[:k].itemsize + V[:k, :].size * V[:k, :].itemsize + U[:k, :].size * U[:k, :].itemsize
    compressed_size2 = Sk.size * Sk.itemsize + V[:k, :].size * V[:k, :].itemsize + U[:k, :].size * U[:k, :].itemsize
    original_text = "Original image uses " + str(original_size) + " bytes"

    st.write("Original image uses", original_size, " bytes")
    col1.write(original_text)
    original_text = "Compressed image uses " + str(compressed_size) + " bytes"
    col2.write(original_text)
    "If S is stored as a diagonal (sparse) matrix, the compressed image is ", np.round(compressed_size / original_size * 100, 5), "% of the original image's size"
    "If S is stored as a 1D list, the compressed image is ", np.round(compressed_size2 / original_size * 100, 5), "% of the original image's size"
    st.metric(label="Compression Ratio", value=np.round(compressed_size2 / original_size, 5))








