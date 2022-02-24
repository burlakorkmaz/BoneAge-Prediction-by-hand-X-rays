import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.image as mpimg       
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import mean_absolute_error
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import Sequential
import cv2
from tensorflow.keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D, Dropout, Input, Flatten, Dense, Concatenate, Conv2D, MaxPooling2D

#%%
def data(path):
    
    #loading dataframes
    train_df = pd.read_csv(path)
    
    #appending file extension to id column for both training and testing dataframes
    train_df['id'] = train_df['id'].apply(lambda x: str(x)+'.png')
    
    #mean boneage
    mean_bone_age = train_df['boneage'].mean()
    
    #standard deviation of boneage
    std_bone_age = train_df['boneage'].std() 
    
    #models perform better when features are normalised to have zero mean and unity standard deviation
    #using z score for the training
    train_df['bone_age_z'] = (train_df['boneage'] - mean_bone_age)/(std_bone_age)
    
    return train_df, mean_bone_age, std_bone_age

# split data
def train_val(train_df, batch_size):
    df_train, df_valid = train_test_split(train_df, test_size = 0.2, random_state = 0)
    batch_train_len = len(df_train)-len(df_train)%batch_size
    batch_val_len = len(df_valid)-len(df_valid)%batch_size
    df_train = df_train[0:batch_train_len]
    df_valid = df_valid[0:batch_val_len]
    
    return df_train, df_valid

# modifying the gender column of the data. replace "male" with "0" and "female" with "1"    
def modify_gender(df_train, df_valid):
    df_train["gender"].replace({'male': 0, 'female': 1},inplace = True)
    df_valid["gender"].replace({'male': 0, 'female': 1},inplace = True)
    return df_train, df_valid

# defining the generator for the concatenated model
def train_generator(model_name, batch_size, img_size, data, img_path):
    if model_name == "vgg":
        from tensorflow.keras.applications.vgg16 import preprocess_input
    elif model_name == "inception_v3":
        from tensorflow.keras.applications.inception_v3 import preprocess_input
    else:    
        from tensorflow.keras.applications.xception import preprocess_input 
    
    if model_name != "cnn":
        train_data_generator = ImageDataGenerator(preprocessing_function = preprocess_input,
                                #featurewise_center=True,
                                #featurewise_std_normalization=True,
                                vertical_flip = True,
                                horizontal_flip= True,
                                rotation_range=20,
                                zoom_range=0.1,
                                width_shift_range=0.05,
                                height_shift_range=0.05)
    else:
        train_data_generator = ImageDataGenerator(rescale=1./255,
                                #featurewise_center=True,
                                #featurewise_std_normalization=True,
                                vertical_flip = True,
                                horizontal_flip= True,
                                rotation_range=20,
                                zoom_range=0.1,
                                width_shift_range=0.05,
                                height_shift_range=0.05)
        
        
    #train data generator
    train_generator = train_data_generator.flow_from_dataframe(
        dataframe = data,
        directory = img_path,
        x_col= 'id',
        y_col= 'bone_age_z',
        batch_size = batch_size,
        seed = 42,
        #shuffle = True,
        class_mode= 'other',
        flip_vertical = True,
        color_mode = 'rgb',
        target_size = (img_size, img_size))
        
    return train_generator
        

def mae_in_months(x_p, y_p):
    '''function to return mae in months'''
    global std_bone_age
    global mean_bone_age
    return mean_absolute_error((std_bone_age*x_p + mean_bone_age), (std_bone_age*y_p + mean_bone_age)) 

def model(model_name, img_size):
    if model_name != "cnn":
        if model_name == "vgg":
            base_model = tf.keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=(img_size,img_size,3))
        
        elif model_name == "inception_v3":
            base_model = tf.keras.applications.InceptionV3(weights='imagenet', 
                                            include_top=False, 
                                            input_shape=(img_size, img_size,3))
    
        else:
            base_model = tf.keras.applications.xception.Xception(input_shape = (img_size, img_size, 3),
                                                      include_top = False,
                                                      weights = 'imagenet')
    
    
        base_model.trainable = True ## Not trainable weights
        model = Sequential()
        model.add(base_model)
        model.add(GlobalMaxPooling2D())
        model.add(Flatten())
        model.add(Dense(60, activation = 'relu'))
        model.add(Dense(10, activation = 'relu'))
        model.add(Dense(1, activation = 'linear')) 
    
    
    else:
        model = Sequential()
        model.add(
            Conv2D(
                filters = 64,
                kernel_size = (7,7),
                strides = 2,
                activation = "relu",
                padding='same',
                input_shape = (img_size, img_size, 3)
            )
        )
        model.add(
            Conv2D(
                filters = 64,
                kernel_size = (7,7),
                strides = 2,
                activation = "relu",
                padding='same',
                input_shape = (img_size, img_size, 3)
            )
        )
            
        model.add(MaxPooling2D(pool_size = 2, strides=None))
        model.add(Dropout(0.2))
        
        model.add(
            Conv2D(
                filters = 128,
                kernel_size = (5,5),
                strides = 2,
                activation = "relu",
                padding='same'
            )
        )
        model.add(
            Conv2D(
                filters = 128,
                kernel_size = (5,5),
                strides = 2,
                activation = "relu",
                padding='same'
            )
        )
        
        model.add(MaxPooling2D(pool_size = 2, strides=None))
        model.add(Dropout(0.2))
        
        model.add(
            Conv2D(
                filters = 256,
                kernel_size = (3, 3),
                strides = 2,
                activation = "relu",
                padding='same'
            )
        )
        model.add(
            Conv2D(
                filters = 256,
                kernel_size = (3, 3),
                strides = 2,
                activation = "relu",
                padding='same'
            )
        )
        model.add(MaxPooling2D(pool_size = 2, strides=None, padding='same'))
        model.add(Dropout(0.2))
        
        model.add(Flatten()),
        model.add(Dense(1024, activation='relu')),
        model.add(Dropout(0.2)),
        model.add(Dense(512, activation='relu')),
        model.add(Dropout(0.2)),
        model.add(Dense(10, activation='relu')),
        model.add(Dense(1, activation='linear'))
    
    return model

#%%

csv_path = '../input/rsna-bone-age/boneage-training-dataset.csv'
img_path = '../input/rsna-bone-age/boneage-training-dataset/boneage-training-dataset'
batch_size = 16
img_size = 256
model_name = "inception_v3" # vgg, inception_v3, xception, cnn

train_df, mean_bone_age, std_bone_age = data(csv_path)
df_train, df_valid = train_val(train_df, batch_size)
df_train, df_valid = modify_gender(df_train, df_valid)

model = model(model_name, img_size)

#compile model
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001, beta_1=0.9, beta_2=0.999,epsilon=1e-7,amsgrad=False,name='Adam')

model.compile(loss ='mae', optimizer= optimizer, metrics = [mae_in_months] )

early_stopping = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience= 15,
                              verbose=2, mode='auto',
                              restore_best_weights = True)

red_lr_plat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=0, mode='auto', cooldown=5, min_delta=0.0001, min_lr=0)

callbacks = [early_stopping, red_lr_plat]

hist = model.fit_generator(train_generator(model_name, batch_size, img_size, df_train, img_path),
        steps_per_epoch=345,
        validation_data=train_generator(model_name, batch_size, img_size, df_valid, img_path),
        validation_steps=1, 
        epochs = 200,
        callbacks= callbacks)

model.save(model_name + "without_gender.h5")




































