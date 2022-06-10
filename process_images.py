import sys
from PIL import Image
import numpy as np
import os

# This utility prepares masks for training for binary masks with uint8 values

def process_image(image_path):
    min = sys.maxsize
    max = -sys.maxsize


    image = Image.open(image_path)
    np_image = np.asarray(image)
    try:
        np_image = np.delete(np_image,3,2)
    except IndexError:
        pass
    if min > np_image.min():
        min = np_image.min()
    if max < np_image.max():
        max = np_image.max()

    np_image = np_image.astype('float32')
    np_image -= min
    np_image /= (max - min)
    image = Image.fromarray(np_image.astype(np.uint8))
    image.save(image_path)

files = os.listdir("TT/masks/")
for file in files:
    process_image("TT/masks/"+file)