import numpy as np
import pandas as pd
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
def generator_train(model_name, batch_size, img_size, data, img_path):
    
    # vgg, inception_v3, xception image date generator with preprocess_input
    if model_name != "cnn":
        if model_name == "vgg":
            from tensorflow.keras.applications.vgg16 import preprocess_input
        elif model_name == "inception_v3":
            from tensorflow.keras.applications.inception_v3 import preprocess_input
        else:    
            from tensorflow.keras.applications.xception import preprocess_input 
            
        train_data_generator = ImageDataGenerator(preprocessing_function = preprocess_input,
                                #featurewise_center=True,
                                #featurewise_std_normalization=True,
                                vertical_flip = True,
                                horizontal_flip= True,
                                rotation_range=20,
                                zoom_range=0.1,
                                width_shift_range=0.05,
                                height_shift_range=0.05)
    
    # cnn image data generator with rescale
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

    
    train_generator = train_data_generator.flow_from_dataframe(
    dataframe = data, 
    directory = img_path,
    x_col= 'id',
    y_col= 'bone_age_z',
    batch_size = batch_size,
    seed = 42,
    shuffle = True,
    class_mode= 'other',
    color_mode = 'rgb',
    target_size = (img_size, img_size))
    
    start = 0
    end = batch_size
    while True:
        a = data['gender'][start:end]
        b = a.array
        start += batch_size
        end += batch_size
        if end == len(data):
            start = 0
            end = batch_size

        X1i = train_generator.next()
        
        yield [X1i[0], b], X1i[1]
            
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
    
        # Model 1
        in1 = Input(shape=(img_size,img_size,3))
        model_one_vgg = base_model(in1)
        model_one_global = GlobalMaxPooling2D()(model_one_vgg)
        #model_one_global = GlobalAveragePooling2D()(model_one_vgg)
        model_one_flat = Flatten()(model_one_global)
        model_one_dense_1 = Dense(60, activation='relu')(model_one_flat)
        model_one_final = Dense(10, activation='relu')(model_one_dense_1)
    
    else:
        in1 = Input(shape=(img_size,img_size,3))
        model_one_cnn1 = Conv2D(
                filters = 64,
                kernel_size = (7,7),
                strides = 2,
                activation = "relu",
                padding='same',
                input_shape = (img_size, img_size, 3)
            )(in1)
        model_one_cnn1_2 = Conv2D(
                filters = 64,
                kernel_size = (7,7),
                strides = 2,
                activation = "relu",
                padding='same',
                input_shape = (img_size, img_size, 3)
            )(model_one_cnn1)
        model_one_globalmax_pooling1 = MaxPooling2D(pool_size = 2, strides=None)(model_one_cnn1_2)
        model_one_dropout1 = Dropout(0.2)(model_one_globalmax_pooling1)
    
        model_one_cnn2 = Conv2D(
                filters = 128,
                kernel_size = (5,5),
                strides = 2,
                activation = "relu",
                padding='same',
                input_shape = (img_size, img_size, 3)
            )(model_one_dropout1)
        model_one_cnn2_2 = Conv2D(
                filters = 128,
                kernel_size = (5,5),
                strides = 2,
                activation = "relu",
                padding='same',
                input_shape = (img_size, img_size, 3)
            )(model_one_cnn2)
        model_one_globalmax_pooling2 = MaxPooling2D(pool_size = 2, strides=None)(model_one_cnn2_2)
        model_one_dropout2 = Dropout(0.2)(model_one_globalmax_pooling2)    
    
        model_one_cnn3 = Conv2D(
                filters = 256,
                kernel_size = (3,3),
                strides = 2,
                activation = "relu",
                padding='same',
                input_shape = (img_size, img_size, 3)
            )(model_one_dropout2)
        model_one_cnn3_2 = Conv2D(
                filters = 256,
                kernel_size = (3,3),
                strides = 2,
                activation = "relu",
                padding='same',
                input_shape = (img_size, img_size, 3)
            )(model_one_cnn3)
        model_one_globalmax_pooling3 = MaxPooling2D(pool_size = 2, strides=None, padding='same')(model_one_cnn3_2)
        model_one_dropout3 = Dropout(0.2)(model_one_globalmax_pooling3)  
    
        model_one_flatten = Flatten()(model_one_dropout3)
        model_one_dense1 = Dense(1024, activation='relu')(model_one_flatten)
        model_one_dropout3 = Dropout(0.2)(model_one_dense1)
        model_one_dense2 = Dense(512, activation='relu')(model_one_dropout3)
        model_one_dropout4 = Dropout(0.2)(model_one_dense2)
        model_one_final = Dense(10, activation='relu')(model_one_dropout4)    
    

    # Model 2
    in2 = Input(shape=(1,))
    model_two_dense_1 = Dense(16, activation='relu')(in2)
    model_two_dense_2 = Dense(8, activation='relu')(model_two_dense_1)

    # Model Final
    model_final_concat = Concatenate(axis=-1)([model_one_final, model_two_dense_2])
    #model_final_dense_1 = Dense(1000, activation='relu')(model_final_concat)
    model_final_dense_2 = Dense(10, activation='relu')(model_final_concat)
    model_final_dense_3 = Dense(1, activation='linear')(model_final_dense_2)
    model = Model(inputs=[in1, in2], outputs=model_final_dense_3)
    
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

# create the model based on the given name
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

hist = model.fit_generator(generator_train(model_name, batch_size, img_size, df_train, img_path),
        steps_per_epoch=345,
        validation_data=generator_train(model_name, batch_size, img_size, df_valid, img_path),
        validation_steps=1, 
        epochs = 200,
        callbacks= callbacks)

model.save(model_name + "_with_gender.h5")

















