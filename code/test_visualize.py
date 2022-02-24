import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.metrics import mean_absolute_error
from tensorflow.keras.models import load_model
import cv2

#%%
img_size = 256
batch_size = 16
test_csv_path = '../input/rsna-bone-age/test_part2.csv'
test_img_path = "..\\input\\rsna-bone-age\\test_part2\\"
data_csv_path = '../input/rsna-bone-age/boneage-training-dataset.csv'

#%%
train_df = pd.read_csv(data_csv_path)
std_bone_age = train_df['boneage'].std() 
mean_bone_age = train_df['boneage'].mean()
def mae_in_months(x_p, y_p):
    '''function to return mae in months'''
    global std_bone_age
    global mean_bone_age
    return mean_absolute_error((std_bone_age*x_p + mean_bone_age), (std_bone_age*y_p + mean_bone_age)) 

test_df = pd.read_csv(test_csv_path)
test_df['id'] = test_df['id'].apply(lambda x: str(x)+'.png')
test_df["gender"].replace({'male': 0, 'female': 1},inplace = True)
test_df["gender"].replace({'male': 0, 'female': 1},inplace = True)
test_df['bone_age_z'] = (test_df['boneage'] - mean_bone_age)/(std_bone_age)
image_ids = test_df['id'].values.tolist()

genders = np.array(test_df['gender'].values.tolist(), dtype=np.float64)
bone_z_scores = np.array(test_df['bone_age_z'].values.tolist(), dtype=np.float64)

#%% -------- Models --------

model1_name = "xception"
model1_type = "0" # "0" = not_concanated, "1" = concanated
model1_path = "without_gender_xception.h5"
model1 = load_model(model1_path,custom_objects={'mae_in_months':mae_in_months})

model2_name = "xception"
model2_type = "1" # "0" = not_concanated, "1" = concanated
model2_path = "with_gender_xception.h5"
model2 = load_model(model2_path,custom_objects={'mae_in_months':mae_in_months})

model3_name = "inception_v3"
model3_type = "0" # "0" = not_concanated, "1" = concanated
model3_path = "without_gender_inception.h5"
model3 = load_model(model3_path,custom_objects={'mae_in_months':mae_in_months})

model4_name = "inception_v3"
model4_type = "1" # "0" = not_concanated, "1" = concanated
model4_path = "with_gender_inception.h5"
model4 = load_model(model4_path,custom_objects={'mae_in_months':mae_in_months})

#%%
model_list = [[model1, model1_name, model1_type, model1_path], [model2, model2_name, model2_type, model2_path],
              [model3, model3_name, model3_type, model3_path], [model4, model4_name, model4_type, model4_path]]

preds = []
for model in model_list:
    images_list = []
    for img_id in image_ids:
        path = test_img_path + img_id
        image_array = cv2.imread(path)
        image_array = cv2.resize(image_array, (img_size, img_size))
        
        if model[1] != "cnn":
            if model[1] == "vgg":
                from tensorflow.keras.applications.vgg16 import preprocess_input
            elif model[1] == "inception_v3":
                from tensorflow.keras.applications.inception_v3 import preprocess_input
            else:    
                from tensorflow.keras.applications.xception import preprocess_input 
            
            image_array = preprocess_input(image_array)
            
        else:
            image_array = image_array / 255.0
            
        images_list.append(image_array)
    
    images_list_copy = images_list.copy()
    images_list = np.array(images_list, dtype=np.float32)

    if model[2] == "1":
        pred = mean_bone_age + std_bone_age*(model[0].predict([images_list, genders], batch_size = batch_size, verbose = True))
    else:
        pred = mean_bone_age + std_bone_age*(model[0].predict(images_list, batch_size = batch_size, verbose = True))
    preds.append(pred)
    
test_months = mean_bone_age + std_bone_age*(bone_z_scores)

for i in range(len(images_list_copy)):
    if test_df['gender'][i] == 0:
        gender = "F"
    else:
        gender = "M"
    
    fig = plt.figure()

    fig.set_figheight(5)
    fig.set_figwidth(10)

    title = "Gender: " + gender + " | Bone Age: " + str(test_df['boneage'][i])
    plt.subplot(121)
    
    plt.title(title)
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    plt.imshow(images_list_copy[i][:,:,0], cmap="bone")
    
    
    plt.subplot(122)
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    
    text = (model_list[0][1] + " " + model_list[0][2] + " pred: " + str(preds[0][i]) + "\n\n" +
            model_list[1][1] + " " + model_list[1][2] + " pred: " + str(preds[1][i]) + "\n\n" +
            model_list[2][1] + " " + model_list[2][2] + " pred: " + str(preds[2][i]) + "\n\n" +
            model_list[3][1] + " " + model_list[3][2] + " pred: " + str(preds[3][i]) + "\n\n\n" +
            "0: without gender  |  1: with gender")
    
    plt.text(0.5,0.5,text,horizontalalignment='center',
     verticalalignment='center', fontsize = 14)
    
    plt.show()