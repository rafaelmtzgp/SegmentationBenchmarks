from keras_segmentation.models.model_utils import transfer_weights
from keras_segmentation.pretrained import pspnet_50_ADE_20K
from keras_segmentation.models.pspnet import pspnet_50


## Run to train the model on the images and masks given.

pretrained_model = pspnet_50_ADE_20K()

new_model = pspnet_50( n_classes=51 )

transfer_weights( pretrained_model , new_model  ) # transfer weights from pre-trained model to your model

new_model.train(
    train_images =  "MADS/images/",
    train_annotations = "MADS/masks/",
    epochs = 5, # Not much needed for fine-tuning
)
new_model.save_weights("checks/checky")