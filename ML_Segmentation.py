# __author_of_the_code_base__ = "Sreenivas Bhattiprolu"
# # https://www.youtube.com/watch?v=blrXmOOWdvQ

import os 
import numpy as np
import cv2
import pandas as pd

# Create DataFrames
df_empty = pd.DataFrame()
df_empty2 = pd.DataFrame()
df = pd.DataFrame()

# Read dataset images
files = os.listdir("path_image/")
for file in files:
    image_path = "path_image/" + file
    img = cv2.imread(image_path)
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#Save original image pixels into a data frame. This is our Feature #1.
    img2 = img.reshape(-1)
    df['Original Image'] = img2
    
    #Generate Gabor features
    num = 1  #To count numbers up in order to give Gabor features a lable in the data frame
    kernels = []
    for theta in range(2):   #Define number of thetas
        theta = theta / 4. * np.pi
        for sigma in (1, 3):  #Sigma with 1 and 3
            for lamda in np.arange(0, np.pi, np.pi / 4):   #Range of wavelengths
                for gamma in (0.05, 0.5):   #Gamma values of 0.05 and 0.5
                
                    
                    gabor_label = 'Gabor' + str(num)  #Label Gabor columns as Gabor1, Gabor2, etc.
    #                print(gabor_label)
                    ksize=9
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                    kernels.append(kernel)
                    #Now filter the image and add values to a new column 
                    fimg = cv2.filter2D(img2, cv2.CV_8UC3, kernel)
                    filtered_img = fimg.reshape(-1)
                    df[gabor_label] = filtered_img  #Labels columns as Gabor1, Gabor2, etc.
                    print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                    num += 1  #Increment for gabor column label
                    
    ########################################
    #Gerate OTHER FEATURES and add them to the data frame
                    
    #CANNY EDGE
    edges = cv2.Canny(img, 100,200)   #Image, min and max values
    edges1 = edges.reshape(-1)
    df['Canny Edge'] = edges1 #Add column to original dataframe
    
    from skimage.filters import roberts, sobel, scharr, prewitt
    
    #ROBERTS EDGE
    edge_roberts = roberts(img)
    edge_roberts1 = edge_roberts.reshape(-1)
    df['Roberts'] = edge_roberts1
    
    #SOBEL
    edge_sobel = sobel(img)
    edge_sobel1 = edge_sobel.reshape(-1)
    df['Sobel'] = edge_sobel1
    
    #SCHARR
    edge_scharr = scharr(img)
    edge_scharr1 = edge_scharr.reshape(-1)
    df['Scharr'] = edge_scharr1
    
    #PREWITT
    edge_prewitt = prewitt(img)
    edge_prewitt1 = edge_prewitt.reshape(-1)
    df['Prewitt'] = edge_prewitt1
    
    #GAUSSIAN with sigma=3
    from scipy import ndimage as nd
    gaussian_img = nd.gaussian_filter(img, sigma=3)
    gaussian_img1 = gaussian_img.reshape(-1)
    df['Gaussian s3'] = gaussian_img1
    
    #GAUSSIAN with sigma=7
    gaussian_img2 = nd.gaussian_filter(img, sigma=7)
    gaussian_img3 = gaussian_img2.reshape(-1)
    df['Gaussian s7'] = gaussian_img3
    
    #MEDIAN with sigma=3
    median_img = nd.median_filter(img, size=3)
    median_img1 = median_img.reshape(-1)
    df['Median s3'] = median_img1
    
    #VARIANCE with size=3
    variance_img = nd.generic_filter(img, np.var, size=3)
    variance_img1 = variance_img.reshape(-1)
    df['Variance s3'] = variance_img1  #Add column to original dataframe
    
    # Save the results in a dataframe
    df_new = pd.concat([df_empty, df])
    df_mean = df_new.groupby(df_new.index)
    df_means = round(df_mean.mean())
    df_empty = df_new
    
######################################                

#Now, add a column in the data frame for the Labels
#For this, we need to import the labeled image
files = os.listdir("path_masks/")
for file in files:
    image_path = "path_masks/" + file
    img = cv2.imread(image_path)
    if img is None:
        continue
    labeled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    labeled_img1 = labeled_img.reshape(-1)
    df['Labels'] = labeled_img1
    df_new2 = pd.concat([df_empty2, df])
    df_mean2 = df_new2.groupby(df_new2.index)
    df_means2 = round(df_mean2.mean())
    df_empty2 = df_new2
    
# print("new",df_means)     
# print("new2",df_means2)

#########################################################

#Define the dependent variable that needs to be predicted (labels)
Y = df_means2["Labels"].values

#Define the independent variables
X = df_means

#Split data into train and test to verify accuracy after fitting the model. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=20)


# Import the model we are using
#RandomForestRegressor is for regression type of problems. 
#For classification we use RandomForestClassifier.
#Both yield similar results except for regressor the result is float
#and for classifier it is an integer. 

from sklearn.ensemble import RandomForestClassifier
# Instantiate model with n number of decision trees
model = RandomForestClassifier(n_estimators = 10, random_state = 42)

###
#SVM
# Train the Linear SVM to compare against Random Forest
#SVM will be slower than Random Forest. 
#Make sure to comment out Fetaure importances lines of code as it does not apply to SVM.
#from sklearn.svm import LinearSVC
#model = LinearSVC(max_iter=100)  #Default of 100 is not converging

# Train the model on training data
model.fit(X_train, y_train)

# verify number of trees used. If not defined above. 
#print('Number of Trees used : ', model.n_estimators)

#STEP 8: TESTING THE MODEL BY PREDICTING ON TEST DATA
#AND CALCULATE THE ACCURACY SCORE
#First test prediction on the training data itself. SHould be good. 
prediction_test_train = model.predict(X_train)

#Test prediction on testing data. 
prediction_test = model.predict(X_test)

#.predict just takes the .predict_proba output and changes everything 
#to 0 below a certain threshold (usually 0.5) respectively to 1 above that threshold.
#In this example we have 4 labels, so the probabilities will for each label stored separately. 
# 
#prediction_prob_test = model.predict_proba(X_test)

#Let us check the accuracy on test data
from sklearn import metrics
#Print the prediction accuracy

#First check the accuracy on training data. This will be higher than test data prediction accuracy.
print ("Accuracy on training data = ", metrics.accuracy_score(y_train, prediction_test_train))
#Check accuracy on test dataset. If this is too low compared to train it indicates overfitting on training data.
print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test))



#This part commented out for SVM testing. Uncomment for random forest. 
#One amazing feature of Random forest is that it provides us info on feature importances
# Get numerical feature importances
#importances = list(model.feature_importances_)

feature_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)
# print(feature_imp)

import pickle

#Save the trained model as pickle string to disk for future use
filename = "ML_model"
pickle.dump(model, open(filename, 'wb'))

#To test the model on future datasets
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict(X)

#Test
files = os.listdir("path_test/")
for file in files:
    image_path = "path_test/" + file
    imgN = cv2.imread(image_path)
    if img is None:
        continue
    imgN = cv2.cvtColor(imgN, cv2.COLOR_BGR2GRAY)

    segmented = result.reshape((imgN.shape))
    
    from matplotlib import pyplot as plt
    # plt.imshow(segmented, cmap ='jet')
    
    # Save Masks
    ret, bw_img = cv2.threshold(segmented, 127, 255, cv2.THRESH_BINARY)
    plt.imsave("path_test/Mask" + file, bw_img, cmap ='jet')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # # 
