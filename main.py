from keras_segmentation.models.model_utils import transfer_weights
from keras_segmentation.pretrained import pspnet_50_ADE_20K
from keras_segmentation.models.pspnet import pspnet_50
import os

# This code loads our trained model and then either evaluates it, generates masks for any given image, or both!
# To segment your own images, change the segment_path to a folder with images, then set generate to 1
# To evaluate on your own images, do the same to eval_image_path and eval_mask_path, then set evaluate to 1

new_model = pspnet_50( n_classes=2)
new_model.load_weights("checks/checks")


## PARAMETERS!
segment_path = "MADS/images/"
eval_image_path="MADS/images/"
eval_mask_path="MADS/masks/"
generate = 0
evaluate = 1
##

# Generate masks on all images of a folder
if generate == 1:
    files = os.listdir("MADS/images/")
    for file in files:
        new_model.predict_segmentation(
            inp="MADS/images/"+file,
            out_fname="MADS/pmasks/"+file,
        )

# Evaluate on all images in a folder with given masks from another folder
if evaluate == 1:
    print(new_model.evaluate_segmentation(
        inp_images_dir="MADS/images/",
        annotations_dir="MADS/masks/",
    ))

## Benchmark results
# MADS: IoU50 = 0.81
# TikTok: IoU50 = 0.35