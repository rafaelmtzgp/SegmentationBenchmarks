# Benchmarks for Deep Learning vs Machine Learning Segmentation methods

This project uses two different methods to perform binary segmentation on human segmentation datasets (MADS https://www.kaggle.com/datasets/tapakah68/segmentation-full-body-mads-dataset and TikTok Dances Dataset https://www.kaggle.com/datasets/tapakah68/segmentation-full-body-tiktok-dancing-dataset)

## Deep Learning Method: PSPNet
Using a pre-trained model from https://github.com/divamgupta/image-segmentation-keras , we finetuned the data using the MADS dataset, and then predicted on the TikTok Dances Dataset. 

## Machine Learning using Random Forest
With random forest and light-based parameters, we obtained masks for the two datasets and evaluated them

## Running the code
**main.py** has the evaluation and mask generation code. Image Segmentation Keras is included for ease of use,  you will still need to have Keras and Tensorflow to run it. You can change the parameters to evaluate or segment your own images.

**train.py** will finetune the PSPNet model from Image Segmentation Keras to the MADS. 

**process_images.py** will prepare the MADS masks for use with Image Segmentation Keras. It will overwrite them, please be careful if you are using the dataset for other purposes.

**process_masks.py** processes the images obtained from the Random Forest method and turns it into a binary masks compatible with main.py. You may need to change the path to your generated images.

**ML_Segmentation.py** has the evaluation and generation of masks using a conventional machine learning method: Random Forest. The libraries required for the use of this code are pandas for the creation of dataframes, OpenCV (cv2) for image analysis, NumPy, and the os module. The included parameters can be removed if desired, as well as the SVM implementation. It is necessary to define the paths to obtain the original image parameters and for training.
