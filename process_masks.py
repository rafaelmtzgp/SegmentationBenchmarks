import numpy as np
from PIL import Image
import os
## Preprocesses RF obtained masks

def procrf(image_path):
    img = Image.open(image_path)
    img = np.asarray(img)
    img = img[:,:,0]
    img[img > 0] = 1
    img = Image.fromarray(np.uint8(img*255))
    img.save(image_path)

files = os.listdir("TT/rfmasks/")
for file in files:
    procrf("TT/rfmasks/"+file)

