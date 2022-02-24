import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.metrics import mean_absolute_error
from tensorflow.keras.models import load_model
import cv2

img_size = 256
batch_size = 16
model_name = "xception"
model_type = "0" # "0" = not_concanated, "1" = concanated
data_csv_path = '../input/rsna-bone-age/boneage-training-dataset.csv'
model_path = "without_gender_xception.h5"
test_csv_path = '../input/rsna-bone-age/boneage-test-dataset.csv'
test_img_path = "..\\input\\rsna-bone-age\\boneage-test-dataset\\boneage-test-dataset\\"


train_df = pd.read_csv(data_csv_path)
std_bone_age = train_df['boneage'].std() 
mean_bone_age = train_df['boneage'].mean()
def mae_in_months(x_p, y_p):
    '''function to return mae in months'''
    global std_bone_age
    global mean_bone_age
    return mean_absolute_error((std_bone_age*x_p + mean_bone_age), (std_bone_age*y_p + mean_bone_age)) 

model = load_model(model_path,custom_objects={'mae_in_months':mae_in_months})

test_df = pd.read_csv(test_csv_path)
test_df['id'] = test_df['id'].apply(lambda x: str(x)+'.png')
test_df["gender"].replace({'male': 0, 'female': 1},inplace = True)
test_df["gender"].replace({'male': 0, 'female': 1},inplace = True)
test_df['bone_age_z'] = (test_df['boneage'] - mean_bone_age)/(std_bone_age)

image_ids = test_df['id'].values.tolist()
images_list = []
for img_id in image_ids:
    path = test_img_path + img_id
    image_array = cv2.imread(path)
    image_array = cv2.resize(image_array, (img_size, img_size))
    
    if model_name != "cnn":
        if model_name == "vgg":
            from tensorflow.keras.applications.vgg16 import preprocess_input
        elif model_name == "inception_v3":
            from tensorflow.keras.applications.inception_v3 import preprocess_input
        else:    
            from tensorflow.keras.applications.xception import preprocess_input 
        
        image_array = preprocess_input(image_array)
        
    else:
        image_array = image_array / 255.0
        
    images_list.append(image_array)
    
images_list = np.array(images_list, dtype=np.float32)
genders = np.array(test_df['gender'].values.tolist(), dtype=np.float64)
bone_z_scores = np.array(test_df['bone_age_z'].values.tolist(), dtype=np.float64)

if model_type == "1":
    pred = mean_bone_age + std_bone_age*(model.predict([images_list, genders], batch_size = batch_size, verbose = True))
else:
    pred = mean_bone_age + std_bone_age*(model.predict(images_list, batch_size = batch_size, verbose = True))
    
test_months = mean_bone_age + std_bone_age*(bone_z_scores)

total_error = 0
for i in range(len(pred)):
    total_error = total_error + abs(pred[i] - test_months[i])
avg_error = total_error/len(pred)
print("AVG MAE(month): ", avg_error)






